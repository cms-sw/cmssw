
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/PatCandidates/interface/PackedCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace pat {
  class PATPackedClusterProducer : public edm::EDProducer {
  public:
    explicit PATPackedClusterProducer(const edm::ParameterSet&);
    ~PATPackedClusterProducer();
    
    virtual void produce(edm::Event&, const edm::EventSetup&);
 
  private:
    bool passSelection_(const reco::CaloCluster& clus,edm::Handle<reco::GsfElectronCollection> eles,edm::Handle<reco::PhotonCollection> phos);
    
    edm::EDGetTokenT<reco::PFClusterCollection>  inputPFClustersToken_;
    edm::EDGetTokenT<reco::CaloClusterCollection>  inputCaloClustersToken_;
    edm::EDGetTokenT<reco::PhotonCollection>  phoToken_;
    edm::EDGetTokenT<reco::GsfElectronCollection>  eleToken_;
    const float maxDeltaR_;

  };
}

pat::PATPackedClusterProducer::PATPackedClusterProducer(const edm::ParameterSet& iConfig) :
  inputPFClustersToken_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("inputClusters"))),
  inputCaloClustersToken_(consumes<reco::CaloClusterCollection>(iConfig.getParameter<edm::InputTag>("inputClusters"))),
  phoToken_(consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("phos"))),
  eleToken_(consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("eles"))),
  maxDeltaR_(iConfig.getParameter<double>("maxDR"))
{
  produces< std::vector<pat::PackedCluster> > ();
}

pat::PATPackedClusterProducer::~PATPackedClusterProducer() {}

void pat::PATPackedClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  
  edm::Handle<reco::PFClusterCollection> inputPFClusters;
  iEvent.getByToken(inputPFClustersToken_,inputPFClusters);

  edm::Handle<reco::CaloClusterCollection> inputCaloClusters;
  iEvent.getByToken(inputCaloClustersToken_,inputCaloClusters);

  edm::Handle<reco::PhotonCollection> phos;
  iEvent.getByToken(phoToken_,phos);
 
  edm::Handle<reco::GsfElectronCollection> eles;
  iEvent.getByToken(eleToken_,eles);

  std::auto_ptr< std::vector<pat::PackedCluster> > outputClustersPtr( new std::vector<pat::PackedCluster> );
  if(inputPFClusters.isValid()){
    for(const auto& clus : *inputPFClusters){
      //     std::cout <<"clus before "<<clus.energy()*sin(clus.position().theta())<<" eta "<<clus.eta()<<" phi "<<clus.phi()<<std::endl;
      if(passSelection_(clus,eles,phos)) outputClustersPtr->push_back(clus);
      //  std::cout <<"clus after "<<outputClustersPtr->back().et()<<" eta "<<outputClustersPtr->back().eta()<<" phi "<<outputClustersPtr->back().phi()<<std::endl;
    }
  }else{
    for(const auto& clus : *inputCaloClusters){
      if(passSelection_(clus,eles,phos)) outputClustersPtr->push_back(clus);
    } 
  }
 
  iEvent.put(outputClustersPtr);
 

}

bool pat::PATPackedClusterProducer::passSelection_(const reco::CaloCluster& clus,edm::Handle<reco::GsfElectronCollection> eles,edm::Handle<reco::PhotonCollection> phos)
{
  if(maxDeltaR_<0) return true;
  const float maxDR2 =maxDeltaR_*maxDeltaR_;
  const float clusEta = clus.eta();
  const float clusPhi = clus.phi();
  if(eles.isValid()){
    for(const auto& ele : *eles){
      if(reco::deltaR2(ele.superCluster()->eta(),ele.superCluster()->phi(),clusEta,clusPhi)< maxDR2) return true;
    }
  }
  if(phos.isValid()){
    for(const auto& pho : *phos){
      if(reco::deltaR2(pho.superCluster()->eta(),pho.superCluster()->phi(),clusEta,clusPhi)< maxDR2) return true;
    }
  }
  return false;
}

using pat::PATPackedClusterProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedClusterProducer);
