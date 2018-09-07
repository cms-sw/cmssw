#ifndef __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class HGCalLayerClusterProducer : public edm::stream::EDProducer<> {
 public:
  HGCalLayerClusterProducer(const edm::ParameterSet&);
  ~HGCalLayerClusterProducer() override { }
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token;

  reco::CaloCluster::AlgoId algoId;

  std::unique_ptr<HGCalImagingAlgo> algo;
  bool doSharing;
  std::string detector;

  HGCalImagingAlgo::VerbosityLevel verbosity;
};

DEFINE_FWK_MODULE(HGCalLayerClusterProducer);

HGCalLayerClusterProducer::HGCalLayerClusterProducer(const edm::ParameterSet &ps) :
  algoId(reco::CaloCluster::undefined),
  doSharing(ps.getParameter<bool>("doSharing")),
  detector(ps.getParameter<std::string >("detector")), // one of EE, FH, BH or "all"
  verbosity((HGCalImagingAlgo::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity",3)){
  double ecut = ps.getParameter<double>("ecut");
  std::vector<double> vecDeltas = ps.getParameter<std::vector<double> >("deltac");
  double kappa = ps.getParameter<double>("kappa");
  std::vector<double> dEdXweights = ps.getParameter<std::vector<double> >("dEdXweights");
  std::vector<double> thicknessCorrection = ps.getParameter<std::vector<double> >("thicknessCorrection");
  std::vector<double> fcPerMip = ps.getParameter<std::vector<double> >("fcPerMip");
  double fcPerEle = ps.getParameter<double>("fcPerEle");
  std::vector<double> nonAgedNoises = ps.getParameter<edm::ParameterSet>("noises").getParameter<std::vector<double> >("values");
  double noiseMip = ps.getParameter<edm::ParameterSet>("noiseMip").getParameter<double>("value");
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


  produces<std::vector<reco::BasicCluster> >();
  produces<std::vector<reco::BasicCluster> >("sharing");

}


void HGCalLayerClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalLayerClusters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "all");
  desc.add<bool>("doSharing", false);
  desc.add<std::vector<double>>("deltac", {
    2.0,
    2.0,
    5.0,
  });
  desc.add<bool>("dependSensor", true);
  desc.add<double>("ecut", 3.0);
  desc.add<double>("kappa", 9.0);
  desc.addUntracked<unsigned int>("verbosity", 3);
  desc.add<edm::InputTag>("HGCEEInput", edm::InputTag("HGCalRecHit","HGCEERecHits"));
  desc.add<edm::InputTag>("HGCFHInput", edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
  desc.add<edm::InputTag>("HGCBHInput", edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
  desc.add<std::vector<double>>("dEdXweights",{});
  desc.add<std::vector<double>>("thicknessCorrection",{});
  desc.add<std::vector<double>>("fcPerMip",{});
  desc.add<double>("fcPerEle",0.0);
  edm::ParameterSetDescription descNestedNoises;
  descNestedNoises.add<std::vector<double> >("values", {});
  desc.add<edm::ParameterSetDescription>("noises", descNestedNoises);
  edm::ParameterSetDescription descNestedNoiseMIP;
  descNestedNoiseMIP.add<double>("value", 0 );
  desc.add<edm::ParameterSetDescription>("noiseMip", descNestedNoiseMIP);
  descriptions.add("hgcalLayerClusters", desc);

}


void HGCalLayerClusterProducer::produce(edm::Event& evt,
				       const edm::EventSetup& es) {

  edm::Handle<HGCRecHitCollection> ee_hits;
  edm::Handle<HGCRecHitCollection> fh_hits;
  edm::Handle<HGCRecHitCollection> bh_hits;


  std::unique_ptr<std::vector<reco::BasicCluster> > clusters( new std::vector<reco::BasicCluster> ),
    clusters_sharing( new std::vector<reco::BasicCluster> );

  algo->reset();

  algo->getEventSetup(es);

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
}

#endif
