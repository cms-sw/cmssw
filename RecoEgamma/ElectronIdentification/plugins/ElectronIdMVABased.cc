// -*- C++ -*-
//
// Package:    ElectronIdMVABased
// Class:      ElectronIdMVABased
//
/**\class ElectronIdMVABased ElectronIdMVABased.cc MyAnalyzer/ElectronIdMVABased/src/ElectronIdMVABased.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Zablocki Jakub
//         Created:  Thu Feb  9 10:47:50 CST 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
// class declaration
//

using namespace std;
using namespace reco;

namespace gsfidhelper {
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet& config) {
      std::vector<std::string> mvaWeightFileEleID = 
        config.getParameter<std::vector<std::string> >("HZZmvaWeightFile");
      ElectronMVAEstimator::Configuration cfg;
      cfg.vweightsfiles = mvaWeightFileEleID;
      mvaID_.reset( new ElectronMVAEstimator(cfg) );
    }
    std::unique_ptr<const ElectronMVAEstimator> mvaID_;
  };
}

class ElectronIdMVABased : public edm::stream::EDFilter< edm::GlobalCache<gsfidhelper::HeavyObjectCache> > {
public:
  explicit ElectronIdMVABased(const edm::ParameterSet&, const gsfidhelper::HeavyObjectCache*);
  ~ElectronIdMVABased();
  
  
  static std::unique_ptr<gsfidhelper::HeavyObjectCache> 
  initializeGlobalCache( const edm::ParameterSet& conf ) {
    return std::unique_ptr<gsfidhelper::HeavyObjectCache>(new gsfidhelper::HeavyObjectCache(conf));
  }
  
  static void globalEndJob(gsfidhelper::HeavyObjectCache const* ) {
  }
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::VertexCollection> vertexToken;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken;
  std::vector<string> mvaWeightFileEleID;
  string path_mvaWeightFileEleID;
  double thresholdBarrel;
  double thresholdEndcap;
  double thresholdIsoBarrel;
  double thresholdIsoEndcap;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronIdMVABased::ElectronIdMVABased(const edm::ParameterSet& iConfig, const gsfidhelper::HeavyObjectCache*) {
  vertexToken = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexTag"));
  electronToken = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronTag"));
  thresholdBarrel = iConfig.getParameter<double>("thresholdBarrel");
  thresholdEndcap = iConfig.getParameter<double>("thresholdEndcap");
  thresholdIsoBarrel = iConfig.getParameter<double>("thresholdIsoDR03Barrel");
  thresholdIsoEndcap = iConfig.getParameter<double>("thresholdIsoDR03Endcap");
  
  produces<reco::GsfElectronCollection>();
}


ElectronIdMVABased::~ElectronIdMVABased()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool ElectronIdMVABased::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  constexpr double etaEBEE = 1.485;
  
  std::auto_ptr<reco::GsfElectronCollection> mvaElectrons(new reco::GsfElectronCollection);
  
  Handle<reco::VertexCollection>  vertexCollection;
  iEvent.getByToken(vertexToken, vertexCollection);
  int nVtx = vertexCollection->size();
  
  Handle<reco::GsfElectronCollection> egCollection;
  iEvent.getByToken(electronToken,egCollection);
  const reco::GsfElectronCollection egCandidates = (*egCollection.product());
  for ( reco::GsfElectronCollection::const_iterator egIter = egCandidates.begin(); egIter != egCandidates.end(); ++egIter) {
    double mvaVal = globalCache()->mvaID_->mva( *egIter, nVtx );
    double isoDr03 = egIter->dr03TkSumPt() + egIter->dr03EcalRecHitSumEt() + egIter->dr03HcalTowerSumEt();
    double eleEta = fabs(egIter->eta());
    if (eleEta <= etaEBEE && mvaVal > thresholdBarrel && isoDr03 < thresholdIsoBarrel) {
      mvaElectrons->push_back( *egIter );
      reco::GsfElectron::MvaOutput myMvaOutput;
      myMvaOutput.mva_Isolated = mvaVal;
      mvaElectrons->back().setMvaOutput(myMvaOutput);
    }
    else if (eleEta > etaEBEE && mvaVal > thresholdEndcap  && isoDr03 < thresholdIsoEndcap) {
      mvaElectrons->push_back( *egIter );
      reco::GsfElectron::MvaOutput myMvaOutput;
      myMvaOutput.mva_Isolated = mvaVal;
      mvaElectrons->back().setMvaOutput(myMvaOutput);
    }
  }
    
  iEvent.put(mvaElectrons);
  
  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronIdMVABased);
