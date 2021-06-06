// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineHLTRcdReader
//
/**\class BeamSpotOnlineHLTRcdReader BeamSpotOnlineHLTRcdReader.cc CondTools/BeamSpot/plugins/BeamSpotOnlineHLTRcdReader.cc

 Description: EDAnalyzer to create a BeamSpotOnlineHLTObjectsRcd payload from a txt file and dump it in a db file

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Francesco Brivio
//         Created:  Tue, 11 Feb 2020 08:39:14 GMT
//
//

// system include files
#include <memory>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

// For ROOT
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>

//
// class declaration
//

class BeamSpotOnlineHLTRcdReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BeamSpotOnlineHLTRcdReader(const edm::ParameterSet&);
  ~BeamSpotOnlineHLTRcdReader() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  struct theBSOfromDB {
    int ls;
    int run;
    float BSx0_;
    float BSy0_;
    float BSz0_;
    float Beamsigmaz_;
    float Beamdxdz_;
    float BeamWidthX_;
    float BeamWidthY_;
    int lastAnalyzedLumi_;
    int lastAnalyzedRun_;
    int lastAnalyzedFill_;
    void init();
  } theBSOfromDB_;

  edm::Service<TFileService> tFileService;
  TTree* bstree_;

  // ----------member data ---------------------------
  edm::ESWatcher<BeamSpotOnlineHLTObjectsRcd> watcher_;
  std::unique_ptr<std::ofstream> output_;
};

//
// constructors and destructor
//
BeamSpotOnlineHLTRcdReader::BeamSpotOnlineHLTRcdReader(const edm::ParameterSet& iConfig) : bstree_(nullptr) {
  //now do what ever initialization is needed
  usesResource("TFileService");
  std::string fileName(iConfig.getUntrackedParameter<std::string>("rawFileName"));
  if (!fileName.empty()) {
    output_.reset(new std::ofstream(fileName.c_str()));
    if (!output_->good()) {
      edm::LogError("IOproblem") << "Could not open output file " << fileName << ".";
      output_.reset();
    }
  }
}

BeamSpotOnlineHLTRcdReader::~BeamSpotOnlineHLTRcdReader() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

void BeamSpotOnlineHLTRcdReader::theBSOfromDB::init() {
  float dummy_float = -999.0;
  int dummy_int = -999;

  run = dummy_int;
  ls = dummy_int;
  BSx0_ = dummy_float;
  BSy0_ = dummy_float;
  BSz0_ = dummy_float;
  Beamsigmaz_ = dummy_float;
  Beamdxdz_ = dummy_float;
  BeamWidthX_ = dummy_float;
  BeamWidthY_ = dummy_float;
  lastAnalyzedLumi_ = dummy_int;
  lastAnalyzedRun_ = dummy_int;
  lastAnalyzedFill_ = dummy_int;
}

// ------------ method called for each event  ------------
void BeamSpotOnlineHLTRcdReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::ostringstream output;

  // initialize the ntuple
  theBSOfromDB_.init();

  if (watcher_.check(iSetup)) {  // check for new IOV for this run / LS

    output << " for runs: " << iEvent.id().run() << " - " << iEvent.id().luminosityBlock() << std::endl;

    // Get BeamSpot from EventSetup:
    edm::ESHandle<BeamSpotOnlineObjects> beamhandle;
    iSetup.get<BeamSpotOnlineHLTObjectsRcd>().get(beamhandle);
    const BeamSpotOnlineObjects* mybeamspot = beamhandle.product();

    theBSOfromDB_.run = iEvent.id().run();
    theBSOfromDB_.ls = iEvent.id().luminosityBlock();
    theBSOfromDB_.BSx0_ = mybeamspot->GetX();
    theBSOfromDB_.BSy0_ = mybeamspot->GetY();
    theBSOfromDB_.BSz0_ = mybeamspot->GetZ();
    theBSOfromDB_.Beamsigmaz_ = mybeamspot->GetSigmaZ();
    theBSOfromDB_.Beamdxdz_ = mybeamspot->Getdxdz();
    theBSOfromDB_.BeamWidthX_ = mybeamspot->GetBeamWidthX();
    theBSOfromDB_.BeamWidthY_ = mybeamspot->GetBeamWidthY();
    theBSOfromDB_.lastAnalyzedLumi_ = mybeamspot->GetLastAnalyzedLumi();
    theBSOfromDB_.lastAnalyzedRun_ = mybeamspot->GetLastAnalyzedRun();
    theBSOfromDB_.lastAnalyzedFill_ = mybeamspot->GetLastAnalyzedFill();

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
void BeamSpotOnlineHLTRcdReader::beginJob() {
  bstree_ = tFileService->make<TTree>("BSONtuple", "BeamSpotOnline analyzer ntuple");

  //Tree Branches
  bstree_->Branch("run", &theBSOfromDB_.run, "run/I");
  bstree_->Branch("ls", &theBSOfromDB_.ls, "ls/I");
  bstree_->Branch("BSx0", &theBSOfromDB_.BSx0_, "BSx0/F");
  bstree_->Branch("BSy0", &theBSOfromDB_.BSy0_, "BSy0/F");
  bstree_->Branch("BSz0", &theBSOfromDB_.BSz0_, "BSz0/F");
  bstree_->Branch("Beamsigmaz", &theBSOfromDB_.Beamsigmaz_, "Beamsigmaz/F");
  bstree_->Branch("Beamdxdz", &theBSOfromDB_.Beamdxdz_, "Beamdxdz/F");
  bstree_->Branch("BeamWidthX", &theBSOfromDB_.BeamWidthX_, "BeamWidthX/F");
  bstree_->Branch("BeamWidthY", &theBSOfromDB_.BeamWidthY_, "BeamWidthY/F");
  bstree_->Branch("LastAnalyzedLumi", &theBSOfromDB_.lastAnalyzedLumi_, "LastAnalyzedLumi/I");
  bstree_->Branch("LastAnalyzedRun", &theBSOfromDB_.lastAnalyzedRun_, "LastAnalyzedRun/I");
  bstree_->Branch("LastAnalyzedFill", &theBSOfromDB_.lastAnalyzedFill_, "LastAnalyzedFill/I");
}

// ------------ method called once each job just after ending the event loop  ------------
void BeamSpotOnlineHLTRcdReader::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineHLTRcdReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineHLTRcdReader);
