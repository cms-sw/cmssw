// -*- C++ -*-
#ifndef Fireworks_Electrons_FWTrackDetailView_h
#define Fireworks_Electrons_FWTrackDetailView_h
//
// Package:     Tracks
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.h,v 1.14 2009/09/03 22:14:15 dmytro Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class FWTrackDetailView : public FWDetailView<reco::Track> {

public:
   FWTrackDetailView();
   virtual ~FWTrackDetailView();

   virtual void build (const FWModelId &id, const reco::Track*, TEveWindowSlot*);

private:
   FWTrackDetailView(const FWTrackDetailView&); // stop default
   const FWTrackDetailView& operator=(const FWTrackDetailView&); // stop default
};

#endif
