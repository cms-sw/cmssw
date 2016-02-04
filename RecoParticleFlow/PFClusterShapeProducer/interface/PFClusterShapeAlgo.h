#ifndef _RecoParticleFlow_PFClusterShapeProducer_PFClusterShapeAlgo_h_
#define _RecoParticleFlow_PFClusterShapeProducer_PFClusterShapeAlgo_h_

#include <map>
#include <algorithm>

#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterShapeAssociation.h"

// Colin: corrected include for compiling in 1_7_X
// #include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"


struct RecHitWithFraction
{
  DetId detId;
  double energy;
  math::XYZVector position;
};

class PFClusterShapeAlgo
{
  typedef std::map<DetId, RecHitWithFraction> RecHitMap;
  
  enum Direction { N, NE, E, SE, S, SW, W, NW };
  enum GeomTopoIndex { BARREL = 0, ENDCAP = 1 }; 

 public:

  explicit PFClusterShapeAlgo(bool useFractions, double w0);

  ~PFClusterShapeAlgo();

  reco::ClusterShapeCollection * makeClusterShapes(edm::Handle<reco::PFClusterCollection> clusterHandle, 
						   edm::Handle<reco::PFRecHitCollection>   rechitHandle,
						   const CaloSubdetectorGeometry * barrelGeo_p,
						   const CaloSubdetectorTopology * barrelTop_p,
						   const CaloSubdetectorGeometry * endcapGeo_p,
						   const CaloSubdetectorTopology * endcapTop_p);
  

 private:

  bool useFractions_;
  double w0_;

  unsigned int currentClusterIndex_;
  reco::PFClusterRef currentCluster_p;
  edm::Handle<reco::PFRecHitCollection> currentRecHit_v_p;

  unsigned int topoIndex;
  std::vector<const CaloSubdetectorTopology *> topoVector;
  unsigned int geomIndex;
  std::vector<const CaloSubdetectorGeometry *> geomVector;

  RecHitWithFraction map5x5[5][5];
  math::XYZVector meanPosition_;
  double totalE_;

  Direction eMaxDir; // the direction of the highest-energy 2x2 subcluster

  DetId eMaxId_, e2ndId_;
  double eMax_, e2nd_;
  double e2x2_, e3x3_, e4x4_, e5x5_, e2x5Right_, e2x5Left_, e2x5Top_, e2x5Bottom_, e3x2_, e3x2Ratio_;
  double covEtaEta_, covEtaPhi_, covPhiPhi_;

  reco::ClusterShape makeClusterShape();

  int findPFRHIndexFromDetId(unsigned int id);
  const reco::PFRecHitFraction * getFractionFromDetId(const DetId & id);

  void fill5x5Map();

  void find_eMax_e2nd();

  double addMapEnergies(int etaIndexLow, int etaIndexHigh, int phiIndexLow, int phiIndexHigh);

  void find_e2x2();
  void find_e3x2();
  void find_e3x3();
  void find_e4x4();
  void find_e5x5();

  void find_e2x5Right();  // "Right" == "North" == greater Phi == greater navigator offset
  void find_e2x5Left();  
  void find_e2x5Top();    // "Top"   == "East"  == lesser Eta  == lesser navigator offset
  void find_e2x5Bottom();

  void covariances();
};

#endif
