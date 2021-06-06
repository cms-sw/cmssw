/**_________________________________________________________________
   class:   AlcaBeamSpotProducer.cc
   package: RecoVertex/BeamSpotProducer



   author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
   Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)


   ________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotProducer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotProducer::AlcaBeamSpotProducer(const edm::ParameterSet &iConfig) {
  // get parameter
  write2DB_ = iConfig.getParameter<edm::ParameterSet>("AlcaBeamSpotProducerParameters").getParameter<bool>("WriteToDB");
  runallfitters_ =
      iConfig.getParameter<edm::ParameterSet>("AlcaBeamSpotProducerParameters").getParameter<bool>("RunAllFitters");
  fitNLumi_ = iConfig.getParameter<edm::ParameterSet>("AlcaBeamSpotProducerParameters")
                  .getUntrackedParameter<int>("fitEveryNLumi", -1);
  resetFitNLumi_ = iConfig.getParameter<edm::ParameterSet>("AlcaBeamSpotProducerParameters")
                       .getUntrackedParameter<int>("resetEveryNLumi", -1);
  runbeamwidthfit_ =
      iConfig.getParameter<edm::ParameterSet>("AlcaBeamSpotProducerParameters").getParameter<bool>("RunBeamWidthFit");

  theBeamFitter = new BeamFitter(iConfig, consumesCollector());
  theBeamFitter->resetTrkVector();
  theBeamFitter->resetLSRange();
  theBeamFitter->resetCutFlow();
  theBeamFitter->resetRefTime();
  theBeamFitter->resetPVFitter();

  ftotalevents = 0;
  ftmprun0 = ftmprun = -1;
  countLumi_ = 0;
  beginLumiOfBSFit_ = endLumiOfBSFit_ = -1;

  produces<reco::BeamSpot, edm::Transition::EndLuminosityBlock>("alcaBeamSpot");
}

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotProducer::~AlcaBeamSpotProducer() { delete theBeamFitter; }

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  ftotalevents++;
  theBeamFitter->readEvent(iEvent);
  ftmprun = iEvent.id().run();
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotProducer::beginLuminosityBlock(edm::LuminosityBlock const &lumiSeg, const edm::EventSetup &iSetup) {
  const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
  const std::time_t ftmptime = fbegintimestamp >> 32;

  if (countLumi_ == 0 || (resetFitNLumi_ > 0 && countLumi_ % resetFitNLumi_ == 0)) {
    ftmprun0 = lumiSeg.run();
    ftmprun = ftmprun0;
    beginLumiOfBSFit_ = lumiSeg.luminosityBlock();
    refBStime[0] = ftmptime;
  }

  countLumi_++;
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotProducer::endLuminosityBlock(edm::LuminosityBlock const &lumiSeg, const edm::EventSetup &iSetup) {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotProducer::endLuminosityBlockProduce(edm::LuminosityBlock &lumiSeg, const edm::EventSetup &iSetup) {
  const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
  const std::time_t fendtime = fendtimestamp >> 32;
  refBStime[1] = fendtime;

  endLumiOfBSFit_ = lumiSeg.luminosityBlock();

  if (fitNLumi_ == -1 && resetFitNLumi_ == -1)
    return;

  if (fitNLumi_ > 0 && countLumi_ % fitNLumi_ != 0)
    return;

  theBeamFitter->setFitLSRange(beginLumiOfBSFit_, endLumiOfBSFit_);
  theBeamFitter->setRefTime(refBStime[0], refBStime[1]);
  theBeamFitter->setRun(ftmprun0);

  std::pair<int, int> LSRange = theBeamFitter->getFitLSRange();

  reco::BeamSpot bs;
  if (theBeamFitter->runPVandTrkFitter()) {
    bs = theBeamFitter->getBeamSpot();
    edm::LogInfo("AlcaBeamSpotProducer") << "\n RESULTS OF DEFAULT FIT " << std::endl
                                         << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl
                                         << " for lumi blocks : " << LSRange.first << " - " << LSRange.second
                                         << std::endl
                                         << " lumi counter # " << countLumi_ << std::endl
                                         << bs << std::endl
                                         << "fit done. \n"
                                         << std::endl;
  } else {  // Fill in empty beam spot if beamfit fails
    bs.setType(reco::BeamSpot::Fake);
    edm::LogInfo("AlcaBeamSpotProducer") << "\n Empty Beam spot fit" << std::endl
                                         << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl
                                         << " for lumi blocks : " << LSRange.first << " - " << LSRange.second
                                         << std::endl
                                         << " lumi counter # " << countLumi_ << std::endl
                                         << bs << std::endl
                                         << "fit failed \n"
                                         << std::endl;
  }

  auto result = std::make_unique<reco::BeamSpot>();
  *result = bs;
  lumiSeg.put(std::move(result), std::string("alcaBeamSpot"));

  if (resetFitNLumi_ > 0 && countLumi_ % resetFitNLumi_ == 0) {
    std::vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
    edm::LogInfo("AlcaBeamSpotProducer") << "Total number of tracks accumulated = " << theBSvector.size() << std::endl
                                         << "Reset track collection for beam fit" << std::endl;
    theBeamFitter->resetTrkVector();
    theBeamFitter->resetLSRange();
    theBeamFitter->resetCutFlow();
    theBeamFitter->resetRefTime();
    theBeamFitter->resetPVFitter();
    countLumi_ = 0;
  }
}

DEFINE_FWK_MODULE(AlcaBeamSpotProducer);
