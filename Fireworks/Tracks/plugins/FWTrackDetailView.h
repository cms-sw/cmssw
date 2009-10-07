// -*- C++ -*-
#ifndef Fireworks_Electrons_FWTrackDetailView_h
#define Fireworks_Electrons_FWTrackDetailView_h
//
// Package:     Tracks
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.h,v 1.2 2009/10/07 14:15:29 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class FWTrackHitsDetailView;
class FWGUISubViewArea;

class FWTrackDetailView : public FWDetailView<reco::Track> {

public:
   FWTrackDetailView();
   virtual ~FWTrackDetailView();

   void hideWindow(FWGUISubviewArea*);
   virtual void build (const FWModelId &id, const reco::Track*, TEveWindowSlot*);

protected:
   FWTrackHitsDetailView* m_hitsView;

private:
   FWTrackDetailView(const FWTrackDetailView&); // stop default
   const FWTrackDetailView& operator=(const FWTrackDetailView&); // stop default
};

#endif
