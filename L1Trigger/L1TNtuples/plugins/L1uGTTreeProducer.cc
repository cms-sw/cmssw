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
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
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

class L1uGTTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1uGTTreeProducer(edm::ParameterSet const &);
  ~L1uGTTreeProducer() override = default;

private:
  void beginJob() override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void endJob() override;

private:
  // output file
  edm::Service<TFileService> fs_;

  // pointers to the objects that will be stored as branches within the tree
  GlobalAlgBlk const *results_;

  // tree
  TTree *tree_;

  // EDM input tokens
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> ugtToken_;
  const edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> l1GtMenuToken_;

  // L1 uGT menu
  unsigned long long cache_id_;
};

L1uGTTreeProducer::L1uGTTreeProducer(edm::ParameterSet const &config)
    : results_(nullptr),
      tree_(nullptr),
      ugtToken_(consumes<GlobalAlgBlkBxCollection>(config.getParameter<edm::InputTag>("ugtToken"))),
      l1GtMenuToken_(esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>()),
      cache_id_(0) {
  usesResource(TFileService::kSharedResource);
  // set up the TTree and its branches
  tree_ = fs_->make<TTree>("L1uGTTree", "L1uGTTree");
  tree_->Branch("L1uGT", "GlobalAlgBlk", &results_, 32000, 3);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1uGTTreeProducer::analyze(edm::Event const &event, edm::EventSetup const &setup) {
  unsigned long long id = setup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();
  if (id != cache_id_) {
    cache_id_ = id;
    edm::ESHandle<L1TUtmTriggerMenu> menu;
    menu = setup.getHandle(l1GtMenuToken_);

    for (auto const &keyval : menu->getAlgorithmMap()) {
      std::string const &name = keyval.second.getName();
      unsigned int index = keyval.second.getIndex();
      //std::cerr << fmt::sprintf("bit %4d: %s", index, name) << std::endl;
      tree_->SetAlias(name.c_str(), fmt::sprintf("L1uGT.m_algoDecisionInitial[%d]", index).c_str());
    }
  }

  edm::Handle<GlobalAlgBlkBxCollection> ugt;
  event.getByToken(ugtToken_, ugt);
  if (ugt.isValid() && ugt.product()->size() != 0) {
    results_ = &ugt->at(0, 0);
  } else {
    edm::LogWarning("MissingProduct")
        << "L1uGTTree or L1uGTTestcrateTree GlobalAlgBlkBxCollection not found. Branch will not be filled.\n"
        << "Please note that the L1uGTTestcrateTree is not expected to exist in MC, so this warning can be ignored!"
        << std::endl;
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1uGTTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1uGTTreeProducer::endJob() {}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1uGTTreeProducer);
