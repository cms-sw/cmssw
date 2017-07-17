#ifndef _HIPixelTrackFilter_h_
#define _HIPixelTrackFilter_h_

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace edm { class EventSetup; }

class HIPixelTrackFilter : public ClusterShapeTrackFilter {
public:
	HIPixelTrackFilter(const SiPixelClusterShapeCache *cache, double ptMin, double ptMax, const edm::EventSetup& es,
	                   const reco::VertexCollection *vertices,
	                   double tipMax, double tipMaxTolerance,
	                   double lipMax, double lipMaxTolerance,
	                   double chi2max,
	                   bool useClusterShape);
	virtual ~HIPixelTrackFilter();
	virtual bool operator() (const reco::Track*, const PixelTrackFilterBase::Hits & hits) const override;
private:
	const reco::VertexCollection *theVertices;
	double theTIPMax, theNSigmaTipMaxTolerance;
	double theLIPMax, theNSigmaLipMaxTolerance;
	double theChi2Max, thePtMin;
	bool useClusterShape;
};

#endif
