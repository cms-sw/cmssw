// system include files
#include <memory>

// {fmt} headers
#include <fmt/printf.h>

// ROOT
#include <TTree.h>

// framework and data formats
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "DataFormats/L1TGlobal/interface/AXOL1TLScore.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//

class L1AXOTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1AXOTreeProducer(edm::ParameterSet const &);
  ~L1AXOTreeProducer() override = default;

private:
  void beginJob() override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void endJob() override;

private:
  // output file
  edm::Service<TFileService> fs_;

  // pointers to the objects that will be stored as branches within the tree
  float anomaly_score;
  float anomaly_inputs[AXOL1TLScore::kNInputs];

  // tree
  TTree *tree_;

  // EDM input tokens
  const edm::EDGetTokenT<AXOL1TLScoreBxCollection> scoreToken_;
};

L1AXOTreeProducer::L1AXOTreeProducer(edm::ParameterSet const &config)
    : anomaly_score(0.0f),
      tree_(nullptr),
      scoreToken_(consumes<AXOL1TLScoreBxCollection>(config.getUntrackedParameter<edm::InputTag>("axoscoreToken"))) {
  usesResource(TFileService::kSharedResource);
  // set up the TTree and its branches
  tree_ = fs_->make<TTree>("L1AXOTree", "L1AXOTree");
  tree_->Branch("axo_score", &anomaly_score, "axo_score/F");

  tree_->Branch("axo_inputs", anomaly_inputs, fmt::sprintf("axo_inputs[%d]/F", AXOL1TLScore::kNInputs).c_str());
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1AXOTreeProducer::analyze(edm::Event const &event, edm::EventSetup const &setup) {
  //save axo score
  edm::Handle<AXOL1TLScoreBxCollection> axo;
  event.getByToken(scoreToken_, axo);

  if (axo.isValid()) {
    // Take bx = 0, index = 0 as before
    const AXOL1TLScore &scoreObj = axo->at(0, 0);

    // score
    anomaly_score = scoreObj.getAXOScore();

    // inputs
    const auto &inputs = scoreObj.getInputs();
    for (unsigned int i = 0; i < AXOL1TLScore::kNInputs; ++i) {
      anomaly_inputs[i] = inputs[i];
    }

  } else {
    edm::LogWarning("MissingProduct") << "AXOL1TLScoreBxCollection not found. Branches will not be filled";

    anomaly_score = 0.f;
    for (unsigned int i = 0; i < AXOL1TLScore::kNInputs; ++i) {
      anomaly_inputs[i] = 0.f;
    }
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1AXOTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1AXOTreeProducer::endJob() {}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1AXOTreeProducer);
