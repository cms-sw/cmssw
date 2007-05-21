#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"
#include "RecoBTau/JetTagMVALearning/interface/JetTagMVATrainer.h"

JetTagMVATrainer::JetTagMVATrainer(const edm::ParameterSet &params) :
	jetId(JetFlavourIdentifier(
		params.getParameter<edm::ParameterSet>("jetIdParameters"))),
	tagInfo(params.getParameter<edm::InputTag>("tagInfo")),
	calibrationLabel(params.getParameter<std::string>("calibrationRecord")),
	minPt(params.getParameter<double>("minimumTransverseMomentum")),
	minEta(params.getParameter<double>("minimumPseudoRapidity")),
	maxEta(params.getParameter<double>("maximumPseudoRapidity")),
	signalFlavours(params.getParameter<std::vector<int> >("signalFlavours")),
	ignoreFlavours(params.getParameter<std::vector<int> >("ignoreFlavours"))
{
	std::sort(signalFlavours.begin(), signalFlavours.end());
	std::sort(ignoreFlavours.begin(), ignoreFlavours.end());
}

JetTagMVATrainer::~JetTagMVATrainer()
{
}

bool JetTagMVATrainer::updateComputer(const edm::EventSetup& es)
{
	// retrieve MVAComputer calibration container
	edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer>
								calibHandle;
	es.get<BTauGenericMVAJetTagComputerRcd>().get("trainer", calibHandle);
	const PhysicsTools::Calibration::MVAComputerContainer *calib =
							calibHandle.product();

	// check container for changes
	if (mvaComputer.get() && calib->changed(containerCacheId)) {
		containerCacheId = calib->getCacheId();

		const PhysicsTools::Calibration::MVAComputer *computerCalib = 
						&calib->find(calibrationLabel);

		if (!computerCalib) {
			mvaComputer.reset();
			return false;
		}

		// check container content for changes
		if (computerCalib->changed(computerCacheId))
			mvaComputer.reset();
	}

	if (!mvaComputer.get()) {
		const PhysicsTools::Calibration::MVAComputer *computerCalib = 
						&calib->find(calibrationLabel);

		if (!computerCalib)
			return false;

		// instantiate new MVAComputer with uptodate calibration
		mvaComputer = std::auto_ptr<GenericMVAComputer>(
					new GenericMVAComputer(computerCalib));

		computerCacheId = computerCalib->getCacheId();
	}

	return true;
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
	// check for uptodate MVAComputer
	if (!updateComputer(es))
		return;

	// pass event to JetFlavourIdentifier
	jetId.readEvent(event);

	// retrieve JetTagInfos
	edm::Handle< edm::View<reco::BaseTagInfo> > tagInfoHandle;
	event.getByLabel(tagInfo, tagInfoHandle);

	// cached array containing MVAComputer value list
	std::vector<PhysicsTools::Variable::Value> values;
	values.push_back(PhysicsTools::Variable::Value(
				PhysicsTools::MVATrainer::kTargetId, 0));

	for(edm::View<reco::BaseTagInfo>::const_iterator iter =
		tagInfoHandle->begin(); iter != tagInfoHandle->end(); iter++) {

		double pt = iter->jet()->pt();
		double eta = std::abs(iter->jet()->eta());
		if (pt < minPt || eta < minEta || eta > maxEta)
			continue;

		// identify jet flavours
		JetFlavour jetFlavour =
				jetId.identifyBasedOnPartons(*iter->jet());
		unsigned int flavour = jetFlavour.flavour();

		// do not train with unknown jet flavours
		if (isIgnoreFlavour(flavour))
			continue;

		// is it a b-jet?
		bool target = isSignalFlavour(flavour);

		reco::TaggingVariableList vars = iter->taggingVariables();

		values.resize(1 + vars.size());
		std::vector<PhysicsTools::Variable::Value>::iterator
						insert = values.begin();

		(insert++)->value = target;
		std::copy(mvaComputer->iterator(vars.begin()),
		          mvaComputer->iterator(vars.end()), insert);

		static_cast<PhysicsTools::MVAComputer*>(
					mvaComputer.get())->eval(values);
	}
}
