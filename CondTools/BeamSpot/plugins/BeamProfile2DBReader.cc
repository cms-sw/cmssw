// -*- C++ -*-
//
// Package:    CondTools/BeamProfile2DBReader
// Class:      BeamProfile2DBReader
//
/**\class BeamProfile2DBReader BeamProfile2DBReader.cc CondTools/BeamSpot/plugins/BeamProfile2DBReader.cc

 Description: simple emd::one::EDAnalyzer to retrieve and ntuplize SimBeamSpot data from the conditions database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Francesco Brivio
//         Created:  11 June 2023
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

// For ROOT
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>

#include <sstream>
#include <fstream>

//
// class declaration
//

class BeamProfile2DBReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BeamProfile2DBReader(const edm::ParameterSet&);
  ~BeamProfile2DBReader() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  struct theBSfromDB {
    int run;
    int ls;
    double fX0, fY0, fZ0;
    double fSigmaZ;
    double fbetastar, femittance;
    double fPhi, fAlpha;
    double fTimeOffset;
    void init();
  } theBSfromDB_;

  const edm::ESGetToken<SimBeamSpotObjects, SimBeamSpotObjectsRcd> beamSpotToken_;
  edm::Service<TFileService> tFileService;
  TTree* bstree_;

  // ----------member data ---------------------------
  edm::ESWatcher<SimBeamSpotObjectsRcd> watcher_;
  std::unique_ptr<std::ofstream> output_;
};

//
// constructors and destructor
//
BeamProfile2DBReader::BeamProfile2DBReader(const edm::ParameterSet& iConfig)
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

BeamProfile2DBReader::~BeamProfile2DBReader() = default;

//
// member functions
//

void BeamProfile2DBReader::theBSfromDB::init() {
  float dummy_double = 0.0;
  int dummy_int = 0;

  run = dummy_int;
  ls = dummy_int;
  fX0 = dummy_double;
  fY0 = dummy_double;
  fZ0 = dummy_double;
  fSigmaZ = dummy_double;
  fbetastar = dummy_double;
  femittance = dummy_double;
  fPhi = dummy_double;
  fAlpha = dummy_double;
  fTimeOffset = dummy_double;
}

// ------------ method called for each event  ------------
void BeamProfile2DBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::ostringstream output;

  // initialize the ntuple
  theBSfromDB_.init();

  if (watcher_.check(iSetup)) {  // check for new IOV for this run / LS

    output << " for runs: " << iEvent.id().run() << " - " << iEvent.id().luminosityBlock() << std::endl;

    // Get SimBeamSpot from EventSetup:
    const SimBeamSpotObjects* mybeamspot = &iSetup.getData(beamSpotToken_);

    theBSfromDB_.run = iEvent.id().run();
    theBSfromDB_.ls = iEvent.id().luminosityBlock();
    theBSfromDB_.fX0 = mybeamspot->x();
    theBSfromDB_.fY0 = mybeamspot->y();
    theBSfromDB_.fZ0 = mybeamspot->z();
    theBSfromDB_.fSigmaZ = mybeamspot->sigmaZ();
    theBSfromDB_.fbetastar = mybeamspot->betaStar();
    theBSfromDB_.femittance = mybeamspot->emittance();
    theBSfromDB_.fPhi = mybeamspot->phi();
    theBSfromDB_.fAlpha = mybeamspot->alpha();
    theBSfromDB_.fTimeOffset = mybeamspot->timeOffset();
    bstree_->Fill();
    output << *mybeamspot << std::endl;
  }

  // Final output - either message logger or output file:
  if (output_.get())
    *output_ << output.str();
  else
    edm::LogInfo("") << output.str();
}

// ------------ method called once each job just before starting event loop  ------------
void BeamProfile2DBReader::beginJob() {
  bstree_ = tFileService->make<TTree>("BSNtuple", "SimBeamSpot analyzer ntuple");

  //Tree Branches
  bstree_->Branch("run", &theBSfromDB_.run, "run/I");
  bstree_->Branch("ls", &theBSfromDB_.ls, "ls/I");
  bstree_->Branch("BSx0", &theBSfromDB_.fX0, "BSx0/F");
  bstree_->Branch("BSy0", &theBSfromDB_.fY0, "BSy0/F");
  bstree_->Branch("BSz0", &theBSfromDB_.fZ0, "BSz0/F");
  bstree_->Branch("Beamsigmaz", &theBSfromDB_.fSigmaZ, "Beamsigmaz/F");
  bstree_->Branch("BetaStar", &theBSfromDB_.fbetastar, "BetaStar/F");
  bstree_->Branch("Emittance", &theBSfromDB_.femittance, "Emittance/F");
  bstree_->Branch("Phi", &theBSfromDB_.fPhi, "Phi/F");
  bstree_->Branch("Alpha", &theBSfromDB_.fAlpha, "Alpha/F");
  bstree_->Branch("TimeOffset", &theBSfromDB_.fTimeOffset, "TimeOffset/F");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamProfile2DBReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamProfile2DBReader);
