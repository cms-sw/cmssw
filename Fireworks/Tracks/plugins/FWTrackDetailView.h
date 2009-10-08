// -*- C++ -*-
#ifndef Fireworks_Electrons_FWTrackDetailView_h
#define Fireworks_Electrons_FWTrackDetailView_h
//
// Package:     Tracks
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.h,v 1.3 2009/10/07 19:02:32 amraktad Exp $
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
   virtual void setBackgroundColor(Color_t col);

protected:
   FWTrackHitsDetailView* m_hitsView;

private:
   FWTrackDetailView(const FWTrackDetailView&); // stop default
   const FWTrackDetailView& operator=(const FWTrackDetailView&); // stop default
};

#endif
