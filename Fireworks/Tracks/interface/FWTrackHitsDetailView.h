// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackHitsDetailView

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Rtypes.h"
#include "Fireworks/Core/interface/FWDetailView.h"
class DetIdToMatrix;
class FWModelId;
class TEveWindowSlot;

class FWTrackHitsDetailView: public FWDetailView<reco::Track>{
public:
   FWTrackHitsDetailView();
   virtual ~FWTrackHitsDetailView();

   void build (const FWModelId &id, const reco::Track*, TEveWindowSlot*);
protected:

private:
   FWTrackHitsDetailView(const FWTrackHitsDetailView&); // stop default
   const FWTrackHitsDetailView& operator=(const FWTrackHitsDetailView&); // stop default
};
