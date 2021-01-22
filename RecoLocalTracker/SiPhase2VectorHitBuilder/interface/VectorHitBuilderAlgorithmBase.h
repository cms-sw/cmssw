#ifndef RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitBuilderAlgorithmBase_H
#define RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitBuilderAlgorithmBase_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"

class VectorHitBuilderAlgorithmBase {
public:
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;
  typedef edmNew::DetSet<Phase2TrackerCluster1D> Detset;
  typedef Detset::const_iterator const_iterator;
  typedef edmNew::DetSetVector<VectorHit> output_t;
  typedef std::pair<StackGeomDet, std::vector<Phase2TrackerCluster1D>> StackClusters;

  VectorHitBuilderAlgorithmBase(const edm::ParameterSet&,
                                const TrackerGeometry*,
                                const TrackerTopology*,
                                const ClusterParameterEstimator<Phase2TrackerCluster1D>*);
  virtual ~VectorHitBuilderAlgorithmBase() {}

  //FIXME::ERICA::this should be template, return different collection for different algo used!!
  virtual void run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                   VectorHitCollection& vhAcc,
                   VectorHitCollection& vhRej,
                   edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                   edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej) const = 0;

  virtual void buildVectorHits(VectorHitCollection& vhAcc,
                               VectorHitCollection& vhRej,
                               DetId detIdStack,
                               const StackGeomDet* stack,
                               edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                               const Detset& DSVinner,
                               const Detset& DSVouter,
                               const std::vector<bool>& phase2OTClustersToSkip = std::vector<bool>()) const = 0;

  virtual VectorHit buildVectorHit(const StackGeomDet* stack,
                                   Phase2TrackerCluster1DRef lower,
                                   Phase2TrackerCluster1DRef upper) const = 0;

  double computeParallaxCorrection(const PixelGeomDetUnit*,
                                   const Point3DBase<float, LocalTag>&,
                                   const PixelGeomDetUnit*,
                                   const Point3DBase<float, LocalTag>&) const;

  void printClusters(const edmNew::DetSetVector<Phase2TrackerCluster1D>& clusters) const;
  void printCluster(const GeomDet* geomDetUnit, const Phase2TrackerCluster1D* cluster) const;

  const TrackerGeometry* tkGeom_;
  const TrackerTopology* tkTopo_;
  const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe_;
  unsigned int nMaxVHforeachStack_;
  std::vector<double> barrelCut_;
  std::vector<double> endcapCut_;

private:
  edm::ESInputTag cpeTag_;
};

#endif
