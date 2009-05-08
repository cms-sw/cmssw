#ifndef RecoParticleFlow_PFClusterShapeProducer_PFClusterShapeProducer_h_
#define RecoParticleFlow_PFClusterShapeProducer_PFClusterShapeProducer_h_

// system include files
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

#include "RecoParticleFlow/PFClusterShapeProducer/interface/PFClusterShapeAlgo.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"


class PFClusterShapeProducer : public edm::EDProducer
{
 public:

  explicit PFClusterShapeProducer(const edm::ParameterSet &);

  ~PFClusterShapeProducer();

  virtual void produce(edm::Event & ev, const edm::EventSetup & es);

 private:

  std::string shapesLabel_;
 
  edm::InputTag  inputTagPFClustersECAL_;
  edm::InputTag  inputTagPFRecHitsECAL_;

  PFClusterShapeAlgo * csAlgo_p;

  edm::Handle<reco::PFClusterCollection>
    getClusterCollection(edm::Event & evt);

  edm::Handle<reco::PFRecHitCollection>
    getRecHitCollection(edm::Event & evt);
};

#endif
