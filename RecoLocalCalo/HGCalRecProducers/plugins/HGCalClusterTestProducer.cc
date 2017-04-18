#ifndef __RecoLocalCalo_HGCRecProducers_PFClusterProducer_H__
#define __RecoLocalCalo_HGCRecProducers_PFClusterProducer_H__

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
#include <chrono>
#include <iostream>

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCal3DClustering.h"

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

  edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token;

  reco::CaloCluster::AlgoId algoId;

  std::unique_ptr<HGCalImagingAlgo> algo;
  std::unique_ptr<HGCal3DClustering> multicluster_algo;
  bool doSharing;
  std::string detector;

  HGCalImagingAlgo::VerbosityLevel verbosity;
};

DEFINE_FWK_MODULE(HGCalClusterTestProducer);

HGCalClusterTestProducer::HGCalClusterTestProducer(const edm::ParameterSet &ps) :
  algoId(reco::CaloCluster::undefined),
  doSharing(ps.getParameter<bool>("doSharing")),
  detector(ps.getParameter<std::string >("detector")),              //one of EE, EF or "both"
  verbosity((HGCalImagingAlgo::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity",3)){
  double ecut = ps.getParameter<double>("ecut");
  std::vector<double> vecDeltas = ps.getParameter<std::vector<double> >("deltac");
  double kappa = ps.getParameter<double>("kappa");
  std::vector<double> multicluster_radii = ps.getParameter<std::vector<double> >("multiclusterRadii");
  double minClusters = ps.getParameter<unsigned>("minClusters");
  std::vector<double> dEdXweights = ps.getParameter<std::vector<double> >("dEdXweights");
  std::vector<double> thicknessCorrection = ps.getParameter<std::vector<double> >("thicknessCorrection");
  std::vector<double> fcPerMip = ps.getParameter<std::vector<double> >("fcPerMip");
  double fcPerEle = ps.getParameter<double>("fcPerEle");
  std::vector<double> nonAgedNoises = ps.getParameter<std::vector<double> >("nonAgedNoises");
  double noiseMip = ps.getParameter<double>("noiseMip");
  bool dependSensor = ps.getParameter<bool>("dependSensor");


  if(detector=="all") {
    hits_ee_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEInput"));
    hits_fh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCFHInput"));
    hits_bh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCBHInput"));
    algoId = reco::CaloCluster::hgcal_mixed;
  }else if(detector=="EE") {
    hits_ee_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEInput"));
    algoId = reco::CaloCluster::hgcal_em;
  }else if(detector=="FH") {
    hits_fh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCFHInput"));
    algoId = reco::CaloCluster::hgcal_had;
  } else {
    hits_bh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCBHInput"));
    algoId = reco::CaloCluster::hgcal_had;
  }


  if(doSharing){
    double showerSigma =  ps.getParameter<double>("showerSigma");
    algo = std::make_unique<HGCalImagingAlgo>(vecDeltas, kappa, ecut, showerSigma, algoId, dependSensor, dEdXweights, thicknessCorrection, fcPerMip, fcPerEle, nonAgedNoises, noiseMip, verbosity);
  }else{
    algo = std::make_unique<HGCalImagingAlgo>(vecDeltas, kappa, ecut, algoId, dependSensor, dEdXweights, thicknessCorrection, fcPerMip, fcPerEle, nonAgedNoises, noiseMip, verbosity);
  }

  auto sumes = consumesCollector();

  multicluster_algo = std::make_unique<HGCal3DClustering>(ps, sumes, multicluster_radii, minClusters);

  produces<std::vector<reco::BasicCluster> >();
  produces<std::vector<reco::BasicCluster> >("sharing");

  produces<std::vector<reco::HGCalMultiCluster> >();
  produces<std::vector<reco::HGCalMultiCluster> >("sharing");
}

void HGCalClusterTestProducer::produce(edm::Event& evt,
				       const edm::EventSetup& es) {

  edm::Handle<HGCRecHitCollection> ee_hits;
  edm::Handle<HGCRecHitCollection> fh_hits;
  edm::Handle<HGCRecHitCollection> bh_hits;


  std::unique_ptr<std::vector<reco::BasicCluster> > clusters( new std::vector<reco::BasicCluster> ),
    clusters_sharing( new std::vector<reco::BasicCluster> );

  algo->reset();

  algo->getEventSetup(es);

  multicluster_algo->getEvent(evt);
  multicluster_algo->getEventSetup(es);

  switch(algoId){
  case reco::CaloCluster::hgcal_em:
    evt.getByToken(hits_ee_token,ee_hits);
    algo->populate(*ee_hits);
    break;
  case  reco::CaloCluster::hgcal_had:
    evt.getByToken(hits_fh_token,fh_hits);
    evt.getByToken(hits_bh_token,bh_hits);
    if( fh_hits.isValid() ) {
      algo->populate(*fh_hits);
    } else if ( bh_hits.isValid() ) {
      algo->populate(*bh_hits);
    }
    break;
  case reco::CaloCluster::hgcal_mixed:
    evt.getByToken(hits_ee_token,ee_hits);
    algo->populate(*ee_hits);
    evt.getByToken(hits_fh_token,fh_hits);
    algo->populate(*fh_hits);
    evt.getByToken(hits_bh_token,bh_hits);
    algo->populate(*bh_hits);
    break;
  default:
    break;
  }
  algo->makeClusters();
  *clusters = algo->getClusters(false);
  if(doSharing)
    *clusters_sharing = algo->getClusters(true);

  std::vector<std::string> names;
  names.push_back(std::string("gen"));
  names.push_back(std::string("calo_face"));

  auto clusterHandle = evt.put(std::move(clusters));
  auto clusterHandleSharing = evt.put(std::move(clusters_sharing),"sharing");

  edm::PtrVector<reco::BasicCluster> clusterPtrs, clusterPtrsSharing;
  for( unsigned i = 0; i < clusterHandle->size(); ++i ) {
    edm::Ptr<reco::BasicCluster> ptr(clusterHandle,i);
    clusterPtrs.push_back(ptr);
  }

  if(doSharing){
    for( unsigned i = 0; i < clusterHandleSharing->size(); ++i ) {
      edm::Ptr<reco::BasicCluster> ptr(clusterHandleSharing,i);
      clusterPtrsSharing.push_back(ptr);
    }
  }

  std::unique_ptr<std::vector<reco::HGCalMultiCluster> >
    multiclusters( new std::vector<reco::HGCalMultiCluster> ),
    multiclusters_sharing( new std::vector<reco::HGCalMultiCluster> );

  std::chrono::high_resolution_clock::time_point then = std::chrono::high_resolution_clock::now();
  *multiclusters = multicluster_algo->makeClusters(clusterPtrs);
  if(doSharing)
    *multiclusters_sharing = multicluster_algo->makeClusters(clusterPtrsSharing);
  evt.put(std::move(multiclusters));
  if(doSharing)
    evt.put(std::move(multiclusters_sharing),"sharing");
  std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::milliseconds>(now - then);
  // delta += float (now.tv_usec - then.tv_usec)/1000.;
  std::cout << "Time taken by multiclustering " << time_span.count() << " ms" << std::endl;
}

#endif
