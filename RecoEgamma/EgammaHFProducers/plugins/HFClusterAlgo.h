#ifndef RECOLOCALCALO_HFCLUSTERPRODUCER_HFCLUSTERALGO_H
#define RECOLOCALCALO_HFCLUSTERPRODUCER_HFCLUSTERALGO_H 1

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include <map>
#include <list>

/** \class HFClusterAlgo
 
  * \author K. Klapoetke -- Minnesota
  */
//$Id:HFClusterAlgo.h,v 1.2 2007/09/19 09:52 K. Klapoetke Minnesota

class HFClusterAlgo {
 public:
  HFClusterAlgo(); 

  void setup(double minTowerEnergy, double seedThreshold,double maximumSL,double m_maximumRenergy,bool usePMTflag,bool usePulseflag, bool forcePulseFlagMC, int correctionSet);

  void isMC(bool isMC) { m_isMC=isMC; }

  /** Analyze the hits */
  void clusterize(const HFRecHitCollection& hf, 
		  const CaloGeometry& geom,
		  reco::HFEMClusterShapeCollection& clusters,
		  reco::SuperClusterCollection& SuperClusters);
  

  void resetForRun();

 private:
  friend class CompareHFCompleteHitET;
  friend class CompareHFCore;
 
  double m_minTowerEnergy, m_seedThreshold,m_maximumSL,m_maximumRenergy;
  bool m_usePMTFlag;
  bool m_usePulseFlag,m_forcePulseFlagMC;
  bool m_isMC;
  int m_correctionSet;
  std::vector<double> m_cutByEta;
  std::vector<double> m_correctionByEta;
  std::vector<double> m_seedmnEta;
  std::vector<double> m_seedMXeta;
  std::vector<double> m_seedmnPhi;
  std::vector<double> m_seedMXphi;
  struct HFCompleteHit {
    HcalDetId id;
    double energy, et;
  };
  bool isPMTHit(const HFRecHit& hfr);
  bool makeCluster(const HcalDetId& seedid,
		   const HFRecHitCollection& hf, 
		   const CaloGeometry& geom,
		   reco::HFEMClusterShape& clusShp,
		   reco::SuperCluster& SClus);
};

#endif 
