#ifndef __RecoLocalCalo_HGCRecProducers_HGCalMultiClusterProducer_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalMultiClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

    
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCal3DClustering.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class HGCalMultiClusterProducer : public edm::stream::EDProducer<> {
 public:
  HGCalMultiClusterProducer(const edm::ParameterSet&);
  ~HGCalMultiClusterProducer() override { }

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
    edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token;
    edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
    edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token;
    edm::EDGetTokenT<std::vector<reco::BasicCluster> > clusters_token;
    edm::EDGetTokenT<std::vector<reco::BasicCluster> > clusters_sharing_token;
    std::unique_ptr<HGCal3DClustering> multicluster_algo;
    bool doSharing;
    HGCalImagingAlgo::VerbosityLevel verbosity;
};

DEFINE_FWK_MODULE(HGCalMultiClusterProducer);

HGCalMultiClusterProducer::HGCalMultiClusterProducer(const edm::ParameterSet &ps) :
  doSharing(ps.getParameter<bool>("doSharing")),
  verbosity((HGCalImagingAlgo::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity",3)){
  std::vector<double> multicluster_radii = ps.getParameter<std::vector<double> >("multiclusterRadii");
  double minClusters = ps.getParameter<unsigned>("minClusters");
  clusters_token = consumes<std::vector<reco::BasicCluster> >(ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  clusters_sharing_token = consumes<std::vector<reco::BasicCluster> >(ps.getParameter<edm::InputTag>("HGCLayerClustersSharing"));
  hits_ee_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEInput"));
  hits_fh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCFHInput"));
  hits_bh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCBHInput"));
  auto sumes = consumesCollector();

  multicluster_algo = std::make_unique<HGCal3DClustering>(ps, sumes, multicluster_radii, minClusters);

  produces<std::vector<reco::HGCalMultiCluster> >();
  produces<std::vector<reco::HGCalMultiCluster> >("sharing");
}

void HGCalMultiClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

    edm::Handle<std::vector<reco::BasicCluster> > clusterHandle;
    edm::Handle<std::vector<reco::BasicCluster> > clusterSharingHandle;


    evt.getByToken(clusters_token, clusterHandle);
    if(doSharing) evt.getByToken(clusters_sharing_token, clusterSharingHandle);


    edm::PtrVector<reco::BasicCluster> clusterPtrs, clusterPtrsSharing;
    for( unsigned i = 0; i < clusterHandle->size(); ++i ) {
        edm::Ptr<reco::BasicCluster> ptr(clusterHandle,i);
        clusterPtrs.push_back(ptr);
    }

    if(doSharing){
        for( unsigned i = 0; i < clusterSharingHandle->size(); ++i ) {
          edm::Ptr<reco::BasicCluster> ptr(clusterSharingHandle,i);
          clusterPtrsSharing.push_back(ptr);
        }
    }

    auto multiclusters = std::make_unique<std::vector<reco::HGCalMultiCluster>>();
    auto multiclusters_sharing = std::make_unique<std::vector<reco::HGCalMultiCluster>>();


    multicluster_algo->getEvent(evt);
    multicluster_algo->getEventSetup(es);

  *multiclusters = multicluster_algo->makeClusters(clusterPtrs);
  if(doSharing)
    *multiclusters_sharing = multicluster_algo->makeClusters(clusterPtrsSharing);
  evt.put(std::move(multiclusters));
  if(doSharing)
    evt.put(std::move(multiclusters_sharing),"sharing");

}

#endif
