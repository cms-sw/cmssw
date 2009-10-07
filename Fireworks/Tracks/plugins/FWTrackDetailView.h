// -*- C++ -*-
#ifndef Fireworks_Electrons_FWTrackDetailView_h
#define Fireworks_Electrons_FWTrackDetailView_h
//
// Package:     Tracks
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.h,v 1.1 2009/09/06 12:59:46 dmytro Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class FWTrackHitsDetailView;


class FWTrackDetailView : public FWDetailView<reco::Track> {

public:
   FWTrackDetailView();
   virtual ~FWTrackDetailView();

   virtual void build (const FWModelId &id, const reco::Track*, TEveWindowSlot*);

protected:
   FWTrackHitsDetailView* m_hitsView;

private:
   FWTrackDetailView(const FWTrackDetailView&); // stop default
   const FWTrackDetailView& operator=(const FWTrackDetailView&); // stop default
};

#endif
