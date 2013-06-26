#include <functional>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <map>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"
#include "JetTagMVATrainer.h"

using namespace reco;
using namespace PhysicsTools;

static const AtomicId kJetPt(TaggingVariableTokens[btau::jetPt]);
static const AtomicId kJetEta(TaggingVariableTokens[btau::jetEta]);

JetTagMVATrainer::JetTagMVATrainer(const edm::ParameterSet &params) :
	jetFlavour(params.getParameter<edm::InputTag>("jetFlavourMatching")),
	minPt(params.getParameter<double>("minimumTransverseMomentum")),
	minEta(params.getParameter<double>("minimumPseudoRapidity")),
	maxEta(params.getParameter<double>("maximumPseudoRapidity")),
	setupDone(false),
	jetTagComputer(params.getParameter<std::string>("jetTagComputer")),
	tagInfos(params.getParameter< std::vector<edm::InputTag> >("tagInfos")),
	signalFlavours(params.getParameter<std::vector<int> >("signalFlavours")),
	ignoreFlavours(params.getParameter<std::vector<int> >("ignoreFlavours"))
{
	std::sort(signalFlavours.begin(), signalFlavours.end());
	std::sort(ignoreFlavours.begin(), ignoreFlavours.end());

	std::vector<std::string> calibrationLabels;
	if (params.getParameter<bool>("useCategories")) {
		categorySelector.reset(new TagInfoMVACategorySelector(params));

		calibrationLabels = categorySelector->getCategoryLabels();
	} else {
		std::string calibrationRecord =
			params.getParameter<std::string>("calibrationRecord");

		calibrationLabels.push_back(calibrationRecord);
	}

	computerCache.reset(new GenericMVAComputerCache(calibrationLabels));
}

JetTagMVATrainer::~JetTagMVATrainer()
{
}

void JetTagMVATrainer::setup(const JetTagComputer &computer)
{
	std::vector<std::string> inputLabels(computer.getInputLabels());

	if (inputLabels.empty())
		inputLabels.push_back("tagInfo");

	if (tagInfos.size() != inputLabels.size()) {
		std::string message("VInputTag size mismatch - the following "
		                    "taginfo labels are needed:\n");
		for(std::vector<std::string>::const_iterator iter =
			inputLabels.begin(); iter != inputLabels.end(); ++iter)
			message += "\"" + *iter + "\"\n";
		throw edm::Exception(edm::errors::Configuration) << message;
	}

	setupDone = true;
}

// map helper
namespace {
	struct JetCompare :
		public std::binary_function<edm::RefToBase<Jet>,
		                            edm::RefToBase<Jet>, bool> {
		inline bool operator () (const edm::RefToBase<Jet> &j1,
		                         const edm::RefToBase<Jet> &j2) const
		{ return j1.key() < j2.key(); }
	};
}

struct JetTagMVATrainer::JetInfo {
	JetInfo() : flavour(0)
	{ leptons[0] = leptons[1] = leptons[2] = 0; }

	unsigned int		flavour;
	bool			leptons[3];
	std::vector<int>	tagInfos;
};

static bool isFlavour(int flavour, const std::vector<int> &list)
{
	std::vector<int>::const_iterator pos =
			std::lower_bound(list.begin(), list.end(), flavour);

	return pos != list.end() && *pos == flavour;
}

bool JetTagMVATrainer::isFlavour(const JetInfo &info,
                                 const std::vector<int> &list)
{
	if (::isFlavour(info.flavour, list))
		return true;
	else if (info.flavour < 4)
		return false;

	for(unsigned int i = 1; i <= 3; i++)
		if (info.leptons[i - 1] &&
		    ::isFlavour(info.flavour * 10 + i, list))
			return true;

	return false;
}

bool JetTagMVATrainer::isSignalFlavour(const JetInfo &info) const
{
	return isFlavour(info, signalFlavours);
}

bool JetTagMVATrainer::isIgnoreFlavour(const JetInfo &info) const
{
	return isFlavour(info, ignoreFlavours);
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

	// retrieve JetTagComputer
	edm::ESHandle<JetTagComputer> computerHandle;
	es.get<JetTagComputerRecord>().get(jetTagComputer, computerHandle);
	const GenericMVAJetTagComputer *computer =
			dynamic_cast<const GenericMVAJetTagComputer*>(
						computerHandle.product());
	if (!computer)
		throw cms::Exception("InvalidCast")
			<< "JetTagComputer is not a MVAJetTagComputer "
			   "in JetTagMVATrainer" << std::endl;

	computer->passEventSetup(es);

	// finalize the JetTagMVALearning <-> JetTagComputer glue setup
	if (!setupDone)
		setup(*computer);

	// retrieve TagInfos
	typedef std::map<edm::RefToBase<Jet>, JetInfo, JetCompare> JetInfoMap;
	JetInfoMap jetInfos;

	std::vector< edm::Handle< edm::View<BaseTagInfo> > >
					tagInfoHandles(tagInfos.size());
	unsigned int nTagInfos = tagInfos.size();
	for(unsigned int i = 0; i < nTagInfos; i++) {
		edm::Handle< edm::View<BaseTagInfo> > &tagInfoHandle =
							tagInfoHandles[i];
		event.getByLabel(tagInfos[i], tagInfoHandle);

		int j = 0;
		for(edm::View<BaseTagInfo>::const_iterator iter =
			tagInfoHandle->begin();
				iter != tagInfoHandle->end(); iter++, j++) {

			JetInfo &jetInfo = jetInfos[iter->jet()];
			if (jetInfo.tagInfos.empty()) {
				jetInfo.tagInfos.resize(nTagInfos, -1);
			}

			jetInfo.tagInfos[i] = j;
		}
	}

	// retrieve jet flavours;
	edm::Handle<JetFlavourMatchingCollection> jetFlavourHandle;
	event.getByLabel(jetFlavour, jetFlavourHandle);

	for(JetFlavourMatchingCollection::const_iterator iter =
		jetFlavourHandle->begin();
				iter != jetFlavourHandle->end(); iter++) {

		JetInfoMap::iterator pos =
			jetInfos.find(edm::RefToBase<Jet>(iter->first));
		if (pos != jetInfos.end()) {
			int flavour = iter->second.getFlavour();
			flavour = std::abs(flavour);
			if (flavour < 100) {
				JetFlavour::Leptons leptons =
						iter->second.getLeptons();

				pos->second.flavour = flavour;
				pos->second.leptons[0] = leptons.electron > 0;
				pos->second.leptons[1] = leptons.muon > 0;
				pos->second.leptons[2] = leptons.tau > 0;
			}
		}
	}

	// cached array containing MVAComputer value list
	std::vector<Variable::Value> values;
	values.push_back(Variable::Value(MVATrainer::kTargetId, 0));
	values.push_back(Variable::Value(kJetPt, 0));
	values.push_back(Variable::Value(kJetEta, 0));

	// now loop over the map and compute all JetTags
	for(JetInfoMap::const_iterator iter = jetInfos.begin();
	    iter != jetInfos.end(); iter++) {
		edm::RefToBase<Jet> jet = iter->first;
		const JetInfo &info = iter->second;

		// simple jet filter
		if (jet->pt() < minPt ||
		    std::abs(jet->eta()) < minEta ||
		    std::abs(jet->eta()) > maxEta)
			continue;

		// do not train with unknown jet flavours
		if (isIgnoreFlavour(info))
			continue;

		// is it a b-jet?
		bool target = isSignalFlavour(info);

		// build TagInfos pointer for helper
		std::vector<const BaseTagInfo*> tagInfoPtrs(nTagInfos);
		for(unsigned int i = 0; i < nTagInfos; i++)  {
			if (info.tagInfos[i] < 0)
				continue;

			tagInfoPtrs[i] =
				&tagInfoHandles[i]->at(info.tagInfos[i]);
		}
		JetTagComputer::TagInfoHelper helper(tagInfoPtrs);

		TaggingVariableList variables =
					computer->taggingVariables(helper);

		// retrieve index of computer in case categories are used
		int index = 0;
		if (categorySelector.get()) {
			index = categorySelector->findCategory(variables);
			if (index < 0)
				continue;
		}

		GenericMVAComputer *mvaComputer =
					computerCache->getComputer(index);
		if (!mvaComputer)
			continue;

		// composite full array of MVAComputer values
		values.resize(3 + variables.size());
		std::vector<Variable::Value>::iterator insert = values.begin();

		(insert++)->setValue(target);
		(insert++)->setValue(jet->pt());
		(insert++)->setValue(jet->eta());
		std::copy(mvaComputer->iterator(variables.begin()),
		          mvaComputer->iterator(variables.end()), insert);

		static_cast<MVAComputer*>(mvaComputer)->eval(values);
	}
}
