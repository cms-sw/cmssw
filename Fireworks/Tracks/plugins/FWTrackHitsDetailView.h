// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackHitsDetailView

#include "Rtypes.h"

#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class TGLEmbeddedViewer;
class FWIntValueListener;
namespace reco {
   class Track;
}

class FWTrackHitsDetailView: public FWDetailViewGL<reco::Track>,
                             public CSGActionSupervisor
{
public:
   FWTrackHitsDetailView();
   virtual ~FWTrackHitsDetailView();

   void pickCameraCenter();
   void transparencyChanged(int);
   void rnrLabels();

   virtual void setBackgroundColor(Color_t);

protected:
   TEveElementList*    m_modules;
   TEveElementList*    m_moduleLabels;
   TEveElementList*    m_hits;
   TGSlider*           m_slider;
   FWIntValueListener* m_sliderListener;
  
private:
   FWTrackHitsDetailView(const FWTrackHitsDetailView&); // stop default
   const FWTrackHitsDetailView& operator=(const FWTrackHitsDetailView&); // stop default

   void build (const FWModelId &id, const reco::Track*);
   void setTextInfo (const FWModelId &id, const reco::Track*);
};
