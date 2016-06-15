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

  edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_ef_token;

  reco::CaloCluster::AlgoId algoId;

  HGCalImagingAlgo *algo;
  bool doSharing;
  std::string detector;

  HGCalImagingAlgo::VerbosityLevel verbosity;
  //  edm::EDGetTokenT<std::vector<reco::PFCluster> > hydraTokens[2];
};

DEFINE_FWK_MODULE(HGCalClusterTestProducer);

HGCalClusterTestProducer::HGCalClusterTestProducer(const edm::ParameterSet &ps) :
  algoId(reco::CaloCluster::undefined),
  algo(0),doSharing(ps.getParameter<bool>("doSharing")),
  detector(ps.getParameter<std::string >("detector")),              //one of EE, EF or "both"
  verbosity((HGCalImagingAlgo::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity",3)){
  double ecut = ps.getParameter<double>("ecut");
  double delta_c = ps.getParameter<double>("deltac");
  double kappa = ps.getParameter<double>("kappa");

  
  if(detector=="both"){
    hits_ee_token = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit:HGCEERecHits"));
    hits_ef_token = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit:HGCHEFRecHits"));
    algoId = reco::CaloCluster::hgcal_mixed;
  }else if(detector=="EE"){
    hits_ee_token = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit:HGCEERecHits"));
    algoId = reco::CaloCluster::hgcal_em;
  }else{
    hits_ef_token = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit:HGCHEFRecHits"));
    algoId = reco::CaloCluster::hgcal_had;
  }
  if(doSharing){
    double showerSigma =  ps.getParameter<double>("showerSigma");
    algo = new HGCalImagingAlgo(delta_c, kappa, ecut, showerSigma, 0, algoId, verbosity);
  }else{
    algo = new HGCalImagingAlgo(delta_c, kappa, ecut, 0, algoId, verbosity);
  }

  // hydraTokens[0] = consumes<std::vector<reco::PFCluster> >( edm::InputTag("FakeClusterGen") );
  // hydraTokens[1] = consumes<std::vector<reco::PFCluster> >( edm::InputTag("FakeClusterCaloFace") );

  std::cout << "Constructing HGCalClusterTestProducer" << std::endl;

  produces<std::vector<reco::BasicCluster> >();
  produces<std::vector<reco::BasicCluster> >("sharing");
}

void HGCalClusterTestProducer::produce(edm::Event& evt, 
				       const edm::EventSetup& es) {
  edm::ESHandle<HGCalGeometry> ee_geom;
  es.get<IdealGeometryRecord>().get("HGCalEESensitive",ee_geom);
  edm::ESHandle<HGCalGeometry> ef_geom;
  es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",ef_geom);

  edm::Handle<HGCRecHitCollection> ee_hits;

  edm::Handle<HGCRecHitCollection> ef_hits;


  std::unique_ptr<std::vector<reco::BasicCluster> > clusters( new std::vector<reco::BasicCluster> ), 
    clusters_sharing( new std::vector<reco::BasicCluster> );
  
  algo->reset();
  switch(algoId){
  case reco::CaloCluster::hgcal_em:
    evt.getByToken(hits_ee_token,ee_hits);
    algo->setGeometry(ee_geom.product());
    algo->makeClusters(*ee_hits);
    break;
  case  reco::CaloCluster::hgcal_had:
    evt.getByToken(hits_ef_token,ef_hits);
    algo->setGeometry(ef_geom.product());
    algo->makeClusters(*ef_hits);
    break;
  case reco::CaloCluster::hgcal_mixed:
    evt.getByToken(hits_ee_token,ee_hits);
    algo->setGeometry(ee_geom.product());
    algo->makeClusters(*ee_hits);
    evt.getByToken(hits_ef_token,ef_hits);
    algo->setGeometry(ef_geom.product());
    algo->makeClusters(*ef_hits);
    break;
  default:
    break;
  }
  *clusters = algo->getClusters(false);
  if(doSharing)
    *clusters_sharing = algo->getClusters(true);

  std::cout << "Density based cluster size: " << clusters->size() << std::endl;
  if(doSharing)
    std::cout << "Sharing clusters size     : " << clusters_sharing->size() << std::endl;
  
  //  edm::Handle<std::vector<reco::PFCluster> > hydra[2];
  std::vector<std::string> names;
  names.push_back(std::string("gen"));
  names.push_back(std::string("calo_face"));
  // for( unsigned i = 0 ; i < 2; ++i ) {
  //   evt.getByToken(hydraTokens[i],hydra[i]);
  //   std::cout << "hydra " << names[i] << " size : " << hydra[i]->size() << std::endl;
  // }

  evt.put(std::move(clusters));
  evt.put(std::move(clusters_sharing),"sharing");
}

#endif
