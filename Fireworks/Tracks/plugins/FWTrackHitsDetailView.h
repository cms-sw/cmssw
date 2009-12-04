// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackHitsDetailView

#include "Rtypes.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Fireworks/Core/interface/FWDetailView.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class TGLEmbeddedViewer;
class TGButtonGroup;
class FWIntValueListener;

class FWTrackHitsDetailView: public FWDetailView<reco::Track>,
                             public CSGActionSupervisor
{
public:
   FWTrackHitsDetailView();
   virtual ~FWTrackHitsDetailView();

   void build (const FWModelId &id, const reco::Track*, TEveWindowSlot*);
   void pickCameraCenter();
   void transparencyChanged(int);

   virtual void setBackgroundColor(Color_t);

protected:
   TGLEmbeddedViewer*  m_viewer;
   TEveElementList*    m_modules;
   TGSlider*           m_slider;
   FWIntValueListener* m_sliderListener;

private:
   FWTrackHitsDetailView(const FWTrackHitsDetailView&); // stop default
   const FWTrackHitsDetailView& operator=(const FWTrackHitsDetailView&); // stop default
};
