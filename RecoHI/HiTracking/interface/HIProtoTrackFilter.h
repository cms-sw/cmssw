#ifndef _HIProtoTrackFilter_h_
#define _HIProtoTrackFilter_h_

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm { class ParameterSet; class EventSetup; class Event;}

class HIProtoTrackFilter : public PixelTrackFilter {
public:
	HIProtoTrackFilter(const edm::ParameterSet& ps, const edm::EventSetup& es);
	HIProtoTrackFilter(const edm::ParameterSet& ps);
	virtual ~HIProtoTrackFilter();
	virtual bool operator() (const reco::Track*, const PixelTrackFilter::Hits & hits) const;
	virtual void update(edm::Event& ev);
private:
	double theTIPMax;
	double theChi2Max, thePtMin;
	bool   doVariablePtMin;
	edm::InputTag theBeamSpotTag; 
	edm::InputTag theSiPixelRecHits;
	const reco::BeamSpot *theBeamSpot;
	double theVariablePtMin;

};

#endif
