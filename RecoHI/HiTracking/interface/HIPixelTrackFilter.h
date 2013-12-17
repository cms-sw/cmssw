#ifndef _HIPixelTrackFilter_h_
#define _HIPixelTrackFilter_h_

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm { class ParameterSet; class EventSetup; class Event;}
class TrackerTopology;

class HIPixelTrackFilter : public ClusterShapeTrackFilter {
public:
	HIPixelTrackFilter(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
	virtual ~HIPixelTrackFilter();
	virtual bool operator() (const reco::Track*, const PixelTrackFilter::Hits & hits,
				 const TrackerTopology *tTopo) const;
	virtual void update(const edm::Event& ev, const edm::EventSetup& es) override;
private:
	double theTIPMax, theNSigmaTipMaxTolerance;
	double theLIPMax, theNSigmaLipMaxTolerance;
	double theChi2Max, thePtMin;
	bool useClusterShape;
	edm::InputTag theVertexCollection; 	
	edm::EDGetTokenT<reco::VertexCollection> theVertexCollectionToken;
	const reco::VertexCollection *theVertices;

};

#endif
