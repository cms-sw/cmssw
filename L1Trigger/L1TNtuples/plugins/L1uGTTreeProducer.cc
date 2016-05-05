// system include files
#include <memory>
#include <boost/format.hpp>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

//
// class declaration
//

class L1uGTTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1uGTTreeProducer(edm::ParameterSet const &);
  ~L1uGTTreeProducer();
  
  
private:
  virtual void beginJob() override;
  virtual void analyze(edm::Event const &, edm::EventSetup const &) override;
  virtual void endJob() override;

private:
  // output file
  edm::Service<TFileService> fs_;
  
  // pointers to the objects that will be stored as branches within the tree
  GlobalAlgBlk const * results_;

  // tree
  TTree * tree_;
 
  // EDM input tokens
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> ugt_token_;

  // L1 uGT menu
  unsigned long long cache_id_;
};



L1uGTTreeProducer::L1uGTTreeProducer(edm::ParameterSet const & config) :
  results_(NULL), tree_(NULL),
  ugt_token_( consumes<GlobalAlgBlkBxCollection>(config.getParameter<edm::InputTag>("ugtToken"))),
  cache_id_( 0 )
{
  // set up the TTree and its branches
  tree_ = fs_->make<TTree>("L1uGTTree", "L1uGTTree");
  tree_->Branch("L1uGT", "GlobalAlgBlk", & results_, 32000, 3);
}


L1uGTTreeProducer::~L1uGTTreeProducer()
{
  if (tree_) { delete tree_; tree_ = NULL; }
  //if (results_) { delete results_; results_ = NULL; }  // It seems TTree owns this pointer...
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1uGTTreeProducer::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
  unsigned long long id = setup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();
  if (id != cache_id_) {
    cache_id_ = id;
    edm::ESHandle<L1TUtmTriggerMenu> menu;
    setup.get<L1TUtmTriggerMenuRcd>().get(menu);

    for (auto const & keyval: menu->getAlgorithmMap()) {
      std::string const & name  = keyval.second.getName();
      unsigned int        index = keyval.second.getIndex();
      //std::cerr << (boost::format("bit %4d: %s") % index % name).str() << std::endl;
      tree_->SetAlias(name.c_str(), (boost::format("L1uGT.m_algoDecisionInitial[%d]") % index).str().c_str());
    }
  }

  edm::Handle<GlobalAlgBlkBxCollection> ugt;

  event.getByToken(ugt_token_, ugt);
  results_ = & ugt->at(0, 0);
  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1uGTTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1uGTTreeProducer::endJob() {
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1uGTTreeProducer);
