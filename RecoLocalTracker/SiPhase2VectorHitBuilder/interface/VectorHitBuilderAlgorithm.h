//---------------------------------------------------------------------------
// class VectorHitBuilderAlgorithm
// author: ebrondol,nathera
// date: May, 2015
//---------------------------------------------------------------------------

#ifndef RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitBuilderAlgorithm_H
#define RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitBuilderAlgorithm_H

#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithmBase.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "CommonTools/Statistics/interface/LinearFit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class VectorHitBuilderAlgorithm : public VectorHitBuilderAlgorithmBase {
public:
  VectorHitBuilderAlgorithm(const edm::ParameterSet& conf,
                            const TrackerGeometry* tkGeomProd,
                            const TrackerTopology* tkTopoProd,
                            const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpeProd)
      : VectorHitBuilderAlgorithmBase(conf, tkGeomProd, tkTopoProd, cpeProd){};
  ~VectorHitBuilderAlgorithm() override = default;

  void run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
           VectorHitCollection& vhAcc,
           VectorHitCollection& vhRej,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej) const override;

  //not implemented yet
  bool checkClustersCompatibilityBeforeBuilding(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                                const Detset& theLowerDetSet,
                                                const Detset& theUpperDetSet) const;
  bool checkClustersCompatibility(Local3DPoint& posinner,
                                  Local3DPoint& posouter,
                                  LocalError& errinner,
                                  LocalError& errouter) const;
  struct CurvatureAndPhi {
    float curvature;
    float curvatureError;
    float phi;
  };
  CurvatureAndPhi curvatureANDphi(Global3DPoint gPositionLower,
                                  Global3DPoint gPositionUpper,
                                  GlobalError gErrorLower,
                                  GlobalError gErrorUpper) const;

  void buildVectorHits(VectorHitCollection& vhAcc,
                       VectorHitCollection& vhRej,
                       DetId detIdStack,
                       const StackGeomDet* stack,
                       edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                       const Detset& DSVinner,
                       const Detset& DSVouter,
                       const std::vector<bool>& phase2OTClustersToSkip = std::vector<bool>()) const override;

  VectorHit buildVectorHit(const StackGeomDet* stack,
                           Phase2TrackerCluster1DRef lower,
                           Phase2TrackerCluster1DRef upper) const override;

  void fit2Dzx(const Local3DPoint lpCI,
               const Local3DPoint lpCO,
               const LocalError leCI,
               const LocalError leCO,
               Local3DPoint& pos,
               Local3DVector& dir,
               AlgebraicSymMatrix22& covMatrix,
               double& chi2) const;
  void fit2Dzy(const Local3DPoint lpCI,
               const Local3DPoint lpCO,
               const LocalError leCI,
               const LocalError leCO,
               Local3DPoint& pos,
               Local3DVector& dir,
               AlgebraicSymMatrix22& covMatrix,
               double& chi2) const;

  void fit(float x[2],
           float y[2],
           float sigy[2],
           Local3DPoint& pos,
           Local3DVector& dir,
           AlgebraicSymMatrix22& covMatrix,
           double& chi2) const;
};

#endif
