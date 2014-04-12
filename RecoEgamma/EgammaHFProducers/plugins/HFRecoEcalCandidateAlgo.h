#ifndef HFRECORECALCANDIDATEALGO_H
#define HFRECORECALCANDIDATEALGO_H 1
#include "HFValueStruct.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include <map>
#include <list>

/** \class HFRecoEcalCandidateAlgo
 
  * \author K. Klapoetke -- Minnesota
  */
//$Id:HFRecoEcalCandidateAlgo.h,v 1.1 2007/09/26 09:52 K. Klapoetke Minnesota

class HFRecoEcalCandidateAlgo {
public:
  HFRecoEcalCandidateAlgo(bool correct, 
			  double e9e25Cut,
			  double intercept2DCut,
			  double intercept2DSlope,
			  const std::vector<double>& e1e9Cut,
			  const std::vector<double>& eCOREe9Cut,
			  const std::vector<double>& eSeLCut,
			  const reco::HFValueStruct hfvv);
  
  /** Analyze the hits */
  void produce(const edm::Handle<reco::SuperClusterCollection>& SuperClusters,
	       const reco::HFEMClusterShapeAssociationCollection& AssocShapes,
	       reco::RecoEcalCandidateCollection& RecoECand,
	       int nvtx);
  
  
 private:
  reco::RecoEcalCandidate correctEPosition(const reco::SuperCluster& original, const reco::HFEMClusterShape& shape, int nvtx);
  bool m_correct;
  double m_e9e25Cut;
  double m_intercept2DCut;
  double m_intercept2DSlope;
  double m_e1e9Cuthi;
  double m_eCOREe9Cuthi;
  double m_eSeLCuthi;
  double m_e1e9Cutlo;
  double m_eCOREe9Cutlo;
  double m_eSeLCutlo;
  int m_era;
  bool m_correctForPileup;
  reco::HFValueStruct m_hfvv;
 
};

#endif 
