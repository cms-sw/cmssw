#ifndef __newpf_PFClusterProducer_H__
#define __newpf_PFClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

#include <memory>

#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCalMultiCluster.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class HGCalClusterTestProducer : public edm::stream::EDProducer<> {
 public:    
  HGCalClusterTestProducer(const edm::ParameterSet&);
  ~HGCalClusterTestProducer() { }
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  edm::EDGetTokenT<HGCRecHitCollection> hits_token;
  edm::EDGetTokenT<std::vector<reco::PFCluster> > hydraTokens[2];
};

DEFINE_FWK_MODULE(HGCalClusterTestProducer);

HGCalClusterTestProducer::HGCalClusterTestProducer(const edm::ParameterSet&) {
  hits_token = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit:HGCEERecHits"));
  hydraTokens[0] = consumes<std::vector<reco::PFCluster> >( edm::InputTag("FakeClusterGen") );
  hydraTokens[1] = consumes<std::vector<reco::PFCluster> >( edm::InputTag("FakeClusterCaloFace") );

  std::cout << "Constructing HGCalClusterTestProducer" << std::endl;

  produces<std::vector<reco::BasicCluster> >();
  produces<std::vector<reco::BasicCluster> >("sharing");
}

void HGCalClusterTestProducer::produce(edm::Event& evt, 
				       const edm::EventSetup& es) {
  edm::ESHandle<HGCalGeometry> ee_geom;
  es.get<IdealGeometryRecord>().get("HGCalEESensitive",ee_geom);

  auto verbosity = HGCalImagingAlgo::pDEBUG;

  HGCalImagingAlgo alg_no_sharing(2, // delta_c for NN
				  10., // kappa
				  0.0001,
				  ee_geom.product(),
				  verbosity);
  
  HGCalImagingAlgo alg_sharing(2, // delta_c for NN
			       10., // kappa
			       0.0001,
			       2.8, // mol. rad. layer 15
			       ee_geom.product(),
			       verbosity);

  edm::Handle<HGCRecHitCollection> ee_hits;
  evt.getByToken(hits_token,ee_hits);
  
  std::unique_ptr<std::vector<reco::BasicCluster> > clusters( new std::vector<reco::BasicCluster> ), 
    clusters_sharing( new std::vector<reco::BasicCluster> );
  
  alg_no_sharing.makeClusters(*ee_hits);
  alg_sharing.makeClusters(*ee_hits);

  *clusters = alg_no_sharing.getClusters();
  *clusters_sharing = alg_sharing.getClusters();

  std::cout << "Density based cluster size: " << clusters->size() << std::endl;
  std::cout << "Sharing clusters size     : " << clusters_sharing->size() << std::endl;
  
  edm::Handle<std::vector<reco::PFCluster> > hydra[2];
  std::vector<std::string> names;
  names.push_back(std::string("gen"));
  names.push_back(std::string("calo_face"));
  for( unsigned i = 0 ; i < 2; ++i ) {
    evt.getByToken(hydraTokens[i],hydra[i]);
    std::cout << "hydra " << names[i] << " size : " << hydra[i]->size() << std::endl;
  }

  evt.put(std::move(clusters));
  evt.put(std::move(clusters_sharing),"sharing");
}

#endif
