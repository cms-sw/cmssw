#ifndef _HIPixelTrackFilter_h_
#define _HIPixelTrackFilter_h_

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm { class ParameterSet; class EventSetup; class Event;}
class TrackerTopology;

class HIPixelTrackFilter : public ClusterShapeTrackFilter {
public:
	HIPixelTrackFilter(const edm::ParameterSet& ps, const edm::EventSetup& es);
	virtual ~HIPixelTrackFilter();
	virtual bool operator() (const reco::Track*, const PixelTrackFilter::Hits & hits,
				 const TrackerTopology *tTopo) const;
	virtual void update(edm::Event& ev);
private:
	double theTIPMax, theNSigmaTipMaxTolerance;
	double theLIPMax, theNSigmaLipMaxTolerance;
	double theChi2Max, thePtMin;
	bool useClusterShape;
	edm::InputTag theVertexCollection; 	
	const reco::VertexCollection *theVertices;

};

#endif
