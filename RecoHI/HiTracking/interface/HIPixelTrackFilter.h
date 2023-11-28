#ifndef _HIPixelTrackFilter_h_
#define _HIPixelTrackFilter_h_

#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class HIPixelTrackFilter : public ClusterShapeTrackFilter {
public:
  HIPixelTrackFilter(const SiPixelClusterShapeCache *cache,
                     double ptMin,
                     double ptMax,
                     const TrackerGeometry *tracker,
                     const ClusterShapeHitFilter *shape,
                     const TrackerTopology *ttopo,
                     const reco::VertexCollection *vertices,
                     double tipMax,
                     double tipMaxTolerance,
                     double lipMax,
                     double lipMaxTolerance,
                     double chi2max,
                     bool useClusterShape);
  ~HIPixelTrackFilter() override;
  bool operator()(const reco::Track *, const PixelTrackFilterBase::Hits &hits) const override;

private:
  const reco::VertexCollection *theVertices;
  double theTIPMax, theNSigmaTipMaxTolerance;
  double theLIPMax, theNSigmaLipMaxTolerance;
  double theChi2Max, thePtMin;
  bool useClusterShape;
};

#endif
