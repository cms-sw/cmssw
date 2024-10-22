// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineRecordsReader
//
/**\class BeamSpotOnlineRecordsReader BeamSpotOnlineRecordsReader.cc CondTools/BeamSpot/plugins/BeamSpotOnlineRecordsReader.cc

 Description: EDAnalyzer to read the BeamSpotOnlineHLTObjectsRcd or BeamSpotOnlineLegacyObjectsRcd and dump it into a txt and root file

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

#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

// For ROOT
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>

//
// class declaration
//

class BeamSpotOnlineRecordsReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BeamSpotOnlineRecordsReader(const edm::ParameterSet&);
  ~BeamSpotOnlineRecordsReader() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  template <class Record>
  void dump(const edm::Event&, const edm::EventSetup&, const edm::ESGetToken<BeamSpotOnlineObjects, Record>&);

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

  const edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> hltToken;
  const edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> legacyToken;

  // ----------member data ---------------------------
  bool isHLT_;
  edm::ESWatcher<BeamSpotOnlineHLTObjectsRcd> hlt_watcher_;
  edm::ESWatcher<BeamSpotOnlineLegacyObjectsRcd> legacy_watcher_;
  std::unique_ptr<std::ofstream> output_;
};

//
// constructors and destructor
//
BeamSpotOnlineRecordsReader::BeamSpotOnlineRecordsReader(const edm::ParameterSet& iConfig)
    : bstree_(nullptr), hltToken(esConsumes()), legacyToken(esConsumes()) {
  //now do what ever initialization is needed
  isHLT_ = iConfig.getParameter<bool>("isHLT");
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

BeamSpotOnlineRecordsReader::~BeamSpotOnlineRecordsReader() = default;

//
// member functions
//

void BeamSpotOnlineRecordsReader::theBSOfromDB::init() {
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
void BeamSpotOnlineRecordsReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (isHLT_) {
    if (hlt_watcher_.check(iSetup)) {
      dump<BeamSpotOnlineHLTObjectsRcd>(iEvent, iSetup, hltToken);
    }  // check for new IOV for this run / LS
  } else {
    if (legacy_watcher_.check(iSetup)) {
      dump<BeamSpotOnlineLegacyObjectsRcd>(iEvent, iSetup, legacyToken);
    }  // check for new IOV for this run / LS
  }
}

template <class Record>
void BeamSpotOnlineRecordsReader::dump(const edm::Event& iEvent,
                                       const edm::EventSetup& iSetup,
                                       const edm::ESGetToken<BeamSpotOnlineObjects, Record>& token) {
  std::ostringstream output;
  // initialize the ntuple
  theBSOfromDB_.init();
  output << " for runs: " << iEvent.id().run() << " - " << iEvent.id().luminosityBlock() << std::endl;

  // Get BeamSpot from EventSetup:
  const BeamSpotOnlineObjects* mybeamspot = &iSetup.getData(token);

  theBSOfromDB_.run = iEvent.id().run();
  theBSOfromDB_.ls = iEvent.id().luminosityBlock();
  theBSOfromDB_.BSx0_ = mybeamspot->x();
  theBSOfromDB_.BSy0_ = mybeamspot->y();
  theBSOfromDB_.BSz0_ = mybeamspot->z();
  theBSOfromDB_.Beamsigmaz_ = mybeamspot->sigmaZ();
  theBSOfromDB_.Beamdxdz_ = mybeamspot->dxdz();
  theBSOfromDB_.BeamWidthX_ = mybeamspot->beamWidthX();
  theBSOfromDB_.BeamWidthY_ = mybeamspot->beamWidthY();
  theBSOfromDB_.lastAnalyzedLumi_ = mybeamspot->lastAnalyzedLumi();
  theBSOfromDB_.lastAnalyzedRun_ = mybeamspot->lastAnalyzedRun();
  theBSOfromDB_.lastAnalyzedFill_ = mybeamspot->lastAnalyzedFill();

  bstree_->Fill();

  output << *mybeamspot << std::endl;

  // Final output - either message logger or output file:
  if (output_.get()) {
    *output_ << output.str();
  } else {
    edm::LogInfo("BeamSpotOnlineRecordsReader") << output.str();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void BeamSpotOnlineRecordsReader::beginJob() {
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

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineRecordsReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isHLT", true);
  desc.addUntracked<std::string>("rawFileName", "");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineRecordsReader);
