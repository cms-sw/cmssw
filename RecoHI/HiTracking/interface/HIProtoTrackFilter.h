#ifndef _HIProtoTrackFilter_h_
#define _HIProtoTrackFilter_h_

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm { class ParameterSet; class EventSetup; class Event;}

class HIProtoTrackFilter : public PixelTrackFilter {
public:
	HIProtoTrackFilter(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
	virtual ~HIProtoTrackFilter();
	virtual bool operator() (const reco::Track*, const PixelTrackFilter::Hits & hits) const;
	virtual void update(const edm::Event& ev, const edm::EventSetup& es) override;
private:
	double theTIPMax;
	double theChi2Max, thePtMin;
	bool   doVariablePtMin;
	edm::InputTag theBeamSpotTag; 
	edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;
	edm::EDGetTokenT<SiPixelRecHitCollection> theSiPixelRecHitsToken;
	const reco::BeamSpot *theBeamSpot;
	double theVariablePtMin;

};

#endif
