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

namespace edm {
  class ParameterSet;
  template <typename T>
  class RefGetter;
  class EventSetup;
}  // namespace edm

class VectorHitBuilderAlgorithmBase {
public:
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;
  typedef edmNew::DetSet<Phase2TrackerCluster1D> detset;
  typedef detset::const_iterator const_iterator;
  typedef edmNew::DetSetVector<VectorHit> output_t;
  typedef std::pair<StackGeomDet, std::vector<Phase2TrackerCluster1D>> StackClusters;

  VectorHitBuilderAlgorithmBase(const edm::ParameterSet&);
  virtual ~VectorHitBuilderAlgorithmBase() {}
  void initialize(const edm::EventSetup&);
  void initTkGeom(edm::ESHandle<TrackerGeometry> tkGeomHandle);
  void initTkTopo(edm::ESHandle<TrackerTopology> tkTopoHandle);
  void initCpe(const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpeProd);

  //FIXME::ERICA::this should be template, return different collection for different algo used!!
  virtual void run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                   VectorHitCollectionNew& vhAcc,
                   VectorHitCollectionNew& vhRej,
                   edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                   edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej) = 0;

  virtual std::vector<std::pair<VectorHit, bool>> buildVectorHits(
      const StackGeomDet* stack,
      edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
      const detset& DSVinner,
      const detset& DSVouter,
      const std::vector<bool>& phase2OTClustersToSkip = std::vector<bool>()) = 0;

  virtual VectorHit buildVectorHit(const StackGeomDet* stack,
                                   Phase2TrackerCluster1DRef lower,
                                   Phase2TrackerCluster1DRef upper) = 0;

  double computeParallaxCorrection(const PixelGeomDetUnit*&,
                                   const Point3DBase<float, LocalTag>&,
                                   const PixelGeomDetUnit*&,
                                   const Point3DBase<float, LocalTag>&);

  void printClusters(const edmNew::DetSetVector<Phase2TrackerCluster1D>& clusters);
  void printCluster(const GeomDet* geomDetUnit, const Phase2TrackerCluster1D* cluster);

  void loadDetSetVector(std::map<DetId, std::vector<VectorHit>>& theMap,
                        edmNew::DetSetVector<VectorHit>& theCollection) const;

  const TrackerGeometry* theTkGeom;
  const TrackerTopology* theTkTopo;
  const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe;
  unsigned int nMaxVHforeachStack;
  std::vector<double> barrelCut;
  std::vector<double> endcapCut;

private:
  edm::ESInputTag cpeTag_;

  //  typedef SiStripRecHit2DCollection::FastFiller Collector;
};

#endif
