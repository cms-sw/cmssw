
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

namespace pat {
  class PATPackedClusterProducer : public edm::EDProducer {
  public:
    explicit PATPackedClusterProducer(const edm::ParameterSet&);
    ~PATPackedClusterProducer();
    
    virtual void produce(edm::Event&, const edm::EventSetup&);
 
  private:
    edm::EDGetTokenT<reco::PFClusterCollection>  inputPFClustersToken_;
    edm::EDGetTokenT<reco::CaloClusterCollection>  inputCaloClustersToken_;
  };
}

pat::PATPackedClusterProducer::PATPackedClusterProducer(const edm::ParameterSet& iConfig) :
  inputPFClustersToken_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("inputClusters"))),
  inputCaloClustersToken_(consumes<reco::CaloClusterCollection>(iConfig.getParameter<edm::InputTag>("inputClusters")))  
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
  
  std::auto_ptr< std::vector<pat::PackedCluster> > outputClustersPtr( new std::vector<pat::PackedCluster> );
  if(inputPFClusters.isValid()){
    for(const auto& clus : *inputPFClusters) outputClustersPtr->push_back(clus);
  }else{
    for(const auto& clus : *inputCaloClusters) outputClustersPtr->push_back(clus);
  }
 
  iEvent.put(outputClustersPtr);
 

}



using pat::PATPackedClusterProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedClusterProducer);
