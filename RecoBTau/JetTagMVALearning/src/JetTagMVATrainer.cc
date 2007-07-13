#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <map>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"
#include "RecoBTau/JetTagMVALearning/interface/JetTagMVATrainer.h"

using namespace reco;
using namespace PhysicsTools;

static const AtomicId kJetPt(TaggingVariableTokens[btau::jetPt]);
static const AtomicId kJetEta(TaggingVariableTokens[btau::jetEta]);

JetTagMVATrainer::JetTagMVATrainer(const edm::ParameterSet &params) :
	jetFlavour(params.getParameter<edm::InputTag>("jetFlavourMatching")),
	tagInfo(params.getParameter<edm::InputTag>("tagInfo")),
	minPt(params.getParameter<double>("minimumTransverseMomentum")),
	minEta(params.getParameter<double>("minimumPseudoRapidity")),
	maxEta(params.getParameter<double>("maximumPseudoRapidity")),
	signalFlavours(params.getParameter<std::vector<int> >("signalFlavours")),
	ignoreFlavours(params.getParameter<std::vector<int> >("ignoreFlavours"))
{
	std::sort(signalFlavours.begin(), signalFlavours.end());
	std::sort(ignoreFlavours.begin(), ignoreFlavours.end());

	std::vector<std::string> calibrationLabels;
	if (params.getParameter<bool>("useCategories")) {
		categorySelector = std::auto_ptr<TagInfoMVACategorySelector>(
				new TagInfoMVACategorySelector(params));

		calibrationLabels = categorySelector->getCategoryLabels();
	} else {
		std::string calibrationRecord =
			params.getParameter<std::string>("calibrationRecord");

		calibrationLabels.push_back(calibrationRecord);
	}

	computerCache = std::auto_ptr<GenericMVAComputerCache>(
			new GenericMVAComputerCache(calibrationLabels));
}

JetTagMVATrainer::~JetTagMVATrainer()
{
}

bool JetTagMVATrainer::isSignalFlavour(int flavour) const
{
	std::vector<int>::const_iterator pos =
		std::lower_bound(signalFlavours.begin(), signalFlavours.end(),
		                 flavour);

	return pos != signalFlavours.end() && *pos == flavour;
}

bool JetTagMVATrainer::isIgnoreFlavour(int flavour) const
{
	std::vector<int>::const_iterator pos =
		std::lower_bound(ignoreFlavours.begin(), ignoreFlavours.end(),
		                 flavour);

	return pos != ignoreFlavours.end() && *pos == flavour;
}

void JetTagMVATrainer::analyze(const edm::Event& event,
                               const edm::EventSetup& es)
{
	// retrieve MVAComputer calibration container
	edm::ESHandle<Calibration::MVAComputerContainer> calibHandle;
	es.get<BTauGenericMVAJetTagComputerRcd>().get("trainer", calibHandle);
	const Calibration::MVAComputerContainer *calib = calibHandle.product();

	// check container for changes
	computerCache->update(calib);
	if (computerCache->isEmpty())
		return;

	// retrieve jet flavours;
	edm::Handle<JetFlavourMatchingCollection> jetFlavourHandle;
	event.getByLabel(jetFlavour, jetFlavourHandle);

	typedef std::map<CaloJetRef, unsigned int> Map_t;
	Map_t flavours;
	for(JetFlavourMatchingCollection::const_iterator iter =
		jetFlavourHandle->begin(); iter != jetFlavourHandle->end(); iter++)
		flavours.insert(*iter);

	// retrieve JetTagInfos
	edm::Handle< edm::View<BaseTagInfo> > tagInfoHandle;
	event.getByLabel(tagInfo, tagInfoHandle);

	// cached array containing MVAComputer value list
	std::vector<Variable::Value> values;
	values.push_back(Variable::Value(MVATrainer::kTargetId, 0));
	values.push_back(Variable::Value(kJetPt, 0));
	values.push_back(Variable::Value(kJetEta, 0));

	for(edm::View<BaseTagInfo>::const_iterator iter =
		tagInfoHandle->begin(); iter != tagInfoHandle->end(); iter++) {

		edm::RefToBase<Jet> jet = iter->jet();

		if (jet->pt() < minPt ||
		    std::abs(jet->eta()) < minEta ||
		    std::abs(jet->eta()) > maxEta)
			continue;

		// identify jet flavours
		Map_t::const_iterator pos =
			flavours.find(jet.castTo<CaloJetRef>());
		if (pos == flavours.end())
			continue;

		unsigned int flavour = pos->second;

		// do not train with unknown jet flavours
		if (isIgnoreFlavour(flavour))
			continue;

		// is it a b-jet?
		bool target = isSignalFlavour(flavour);

		TaggingVariableList variables = iter->taggingVariables();

		// retrieve index of computer in case categories are used
		int index = 0;
		if (categorySelector.get()) {
			index = categorySelector->findCategory(variables);
			if (index < 0)
				continue;
		}

		GenericMVAComputer *computer =
					computerCache->getComputer(index);
		if (!computer)
			continue;

		// composite full array of MVAComputer values
		values.resize(3 + variables.size());
		std::vector<Variable::Value>::iterator insert = values.begin();

		(insert++)->value = target;
		(insert++)->value = jet->pt();
		(insert++)->value = jet->eta();
		std::copy(computer->iterator(variables.begin()),
		          computer->iterator(variables.end()), insert);

		static_cast<MVAComputer*>(computer)->eval(values);
	}
}
