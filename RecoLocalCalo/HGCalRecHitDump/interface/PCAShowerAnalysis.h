#ifndef PCAShowerAnalysis_h
#define PCAShowerAnalysis_h

//===================================================================
// Purpose: shower average and axes reconstruction in HGCAL EE 
// Author: Claude Charlot - LLR- Ecole Polytechnique
// 12/2014
// Comment: modified to be used on any cluster, including a set of preselected RecHits by P. Silva (CERN)
//===================================================================

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "RecoLocalCalo/HGCalRecHitDump/interface/SlimmedRecHit.h"

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TPrincipal.h"

class PCAShowerAnalysis
{
  public:

  struct PCASummary_t
  {
    float center_x, center_y, center_z;
    float axis_x,   axis_y,   axis_z;
    float ev_1,     ev_2,     ev_3;
    float sigma_1,  sigma_2,  sigma_3;
  };  

  PCAShowerAnalysis(bool segmented=true, bool logweighting=true, bool debug=false ) ;
  PCASummary_t computeShowerParameters(const reco::CaloCluster &cl,const SlimmedRecHitCollection &recHits);
  ~PCAShowerAnalysis();

private:

  TPrincipal *principal_;

  bool debug_;
  bool logweighting_;
  bool segmented_;
  double entryz_;
  
};
#endif
