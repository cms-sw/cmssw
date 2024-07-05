// -*- C++ -*-
//
// Package:    CondTools/BeamProfileHLLHC2DBReader
// Class:      BeamProfileHLLHC2DBReader
//
/**\class BeamProfileHLLHC2DBReader BeamProfileHLLHC2DBReader.cc CondTools/BeamSpot/plugins/BeamProfileHLLHC2DBReader.cc

 Description: simple edm::one::EDAnalyzer to retrieve and ntuplize SimBeamSpotHLLHC data from the conditions database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Francesco Brivio
//         Created:  11 June 2023
//

// system include files
#include <fstream>
#include <memory>
#include <sstream>

// user include files
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotHLLHCObjects.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotHLLHCObjectsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// For ROOT
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <TTree.h>

//
// class declaration
//

class BeamProfileHLLHC2DBReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BeamProfileHLLHC2DBReader(const edm::ParameterSet&);
  ~BeamProfileHLLHC2DBReader() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  struct TheBSfromDB {
    int run;
    int ls;
    double fMeanX, fMeanY, fMeanZ;
    double fEProton, fCrabFrequency, fRF800;
    double fCrossingAngle, fCrabbingAngleCrossing, fCrabbingAngleSeparation;
    double fBetaCrossingPlane, fBetaSeparationPlane;
    double fHorizontalEmittance, fVerticalEmittance;
    double fBunchLength, fTimeOffset;
    void init();
  } theBSfromDB_;

  const edm::ESGetToken<SimBeamSpotHLLHCObjects, SimBeamSpotHLLHCObjectsRcd> beamSpotToken_;
  edm::Service<TFileService> tFileService;
  TTree* bstree_;

  // ----------member data ---------------------------
  edm::ESWatcher<SimBeamSpotHLLHCObjectsRcd> watcher_;
  std::unique_ptr<std::ofstream> output_;
};

// ------------ constructor  ------------
BeamProfileHLLHC2DBReader::BeamProfileHLLHC2DBReader(const edm::ParameterSet& iConfig)
    : beamSpotToken_(esConsumes()), bstree_(nullptr) {
  //now do what ever initialization is needed
  usesResource("TFileService");
  std::string fileName(iConfig.getUntrackedParameter<std::string>("rawFileName"));
  if (!fileName.empty()) {
    output_ = std::make_unique<std::ofstream>(fileName.c_str());
    if (!output_->good()) {
      edm::LogError("IOproblem") << "Could not open output file " << fileName << ".";
      output_.reset();
    }
  }
}

// ------------ SimBeamSpotHLLHCObjects initialization  ------------
void BeamProfileHLLHC2DBReader::TheBSfromDB::init() {
  float dummy_double = 0.0;
  int dummy_int = 0;

  run = dummy_int;
  ls = dummy_int;
  fMeanX = dummy_double;
  fMeanY = dummy_double;
  fMeanZ = dummy_double;
  fEProton = dummy_double;
  fCrabFrequency = dummy_double;
  fRF800 = dummy_double;
  fCrossingAngle = dummy_double;
  fCrabbingAngleCrossing = dummy_double;
  fCrabbingAngleSeparation = dummy_double;
  fBetaCrossingPlane = dummy_double;
  fBetaSeparationPlane = dummy_double;
  fHorizontalEmittance = dummy_double;
  fVerticalEmittance = dummy_double;
  fBunchLength = dummy_double;
  fTimeOffset = dummy_double;
}

// ------------ method called for each event  ------------
void BeamProfileHLLHC2DBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::ostringstream output;

  // initialize the ntuple
  theBSfromDB_.init();

  if (watcher_.check(iSetup)) {  // check for new IOV for this run / LS

    output << " for runs: " << iEvent.id().run() << " - " << iEvent.id().luminosityBlock() << std::endl;

    // Get SimBeamSpotHLLHCObjects from EventSetup:
    const SimBeamSpotHLLHCObjects* mybeamspot = &iSetup.getData(beamSpotToken_);

    theBSfromDB_.run = iEvent.id().run();
    theBSfromDB_.ls = iEvent.id().luminosityBlock();
    theBSfromDB_.fMeanX = mybeamspot->meanX();
    theBSfromDB_.fMeanY = mybeamspot->meanY();
    theBSfromDB_.fMeanZ = mybeamspot->meanZ();
    theBSfromDB_.fEProton = mybeamspot->eProton();
    theBSfromDB_.fCrabFrequency = mybeamspot->crabFrequency();
    theBSfromDB_.fRF800 = mybeamspot->rf800();
    theBSfromDB_.fCrossingAngle = mybeamspot->crossingAngle();
    theBSfromDB_.fCrabbingAngleCrossing = mybeamspot->crabbingAngleCrossing();
    theBSfromDB_.fCrabbingAngleSeparation = mybeamspot->crabbingAngleSeparation();
    theBSfromDB_.fBetaCrossingPlane = mybeamspot->betaCrossingPlane();
    theBSfromDB_.fBetaSeparationPlane = mybeamspot->betaSeparationPlane();
    theBSfromDB_.fHorizontalEmittance = mybeamspot->horizontalEmittance();
    theBSfromDB_.fVerticalEmittance = mybeamspot->verticalEmittance();
    theBSfromDB_.fBunchLength = mybeamspot->bunchLenght();
    theBSfromDB_.fTimeOffset = mybeamspot->timeOffset();
    bstree_->Fill();
    output << *mybeamspot << std::endl;
  }

  // Final output - either message logger or output file:
  if (output_.get())
    *output_ << output.str();
  else
    edm::LogInfo("BeamProfileHLLHC2DBReader") << output.str();
}

// ------------ method called once each job just before starting event loop  ------------
void BeamProfileHLLHC2DBReader::beginJob() {
  bstree_ = tFileService->make<TTree>("BSNtuple", "SimBeamSpotHLLHC analyzer ntuple");

  //Tree Branches
  bstree_->Branch("run", &theBSfromDB_.run, "run/I");
  bstree_->Branch("ls", &theBSfromDB_.ls, "ls/I");
  bstree_->Branch("MeanX", &theBSfromDB_.fMeanX, "MeanX/F");
  bstree_->Branch("MeanY", &theBSfromDB_.fMeanY, "MeanY/F");
  bstree_->Branch("MeanZ", &theBSfromDB_.fMeanZ, "MeanZ/F");
  bstree_->Branch("EProton", &theBSfromDB_.fEProton, "EProton/F");
  bstree_->Branch("CrabFrequency", &theBSfromDB_.fCrabFrequency, "CrabFrequency/F");
  bstree_->Branch("RF800", &theBSfromDB_.fRF800, "RF800/O");
  bstree_->Branch("CrossingAngle", &theBSfromDB_.fCrossingAngle, "CrossingAngle/F");
  bstree_->Branch("CrabbingAngleCrossing", &theBSfromDB_.fCrabbingAngleCrossing, "CrabbingAngleCrossing/F");
  bstree_->Branch("CrabbingAngleSeparation", &theBSfromDB_.fCrabbingAngleSeparation, "CrabbingAngleSeparation/F");
  bstree_->Branch("BetaCrossingPlane", &theBSfromDB_.fBetaCrossingPlane, "BetaCrossingPlane/F");
  bstree_->Branch("BetaSeparationPlane", &theBSfromDB_.fBetaSeparationPlane, "BetaSeparationPlane/F");
  bstree_->Branch("HorizontalEmittance", &theBSfromDB_.fHorizontalEmittance, "HorizontalEmittance/F");
  bstree_->Branch("VerticalEmittance", &theBSfromDB_.fVerticalEmittance, "VerticalEmittance/F");
  bstree_->Branch("BunchLength", &theBSfromDB_.fBunchLength, "BunchLength/F");
  bstree_->Branch("TimeOffset", &theBSfromDB_.fTimeOffset, "TimeOffset/F");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamProfileHLLHC2DBReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("rawFileName", {});
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamProfileHLLHC2DBReader);
