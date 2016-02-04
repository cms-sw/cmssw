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
   void makeLegend( void );

   void
   addTrackerHits3D( std::vector<TVector3> &points,
		     class TEveElementList *tList,
		     Color_t color, int size );

   void
   addHits( const reco::Track& track,
	    const FWEventItem* iItem,
	    TEveElement* trkList,
	    bool addNearbyHits );
   void
   addModules( const reco::Track& track,
	       const FWEventItem* iItem,
	       TEveElement* trkList,
	       bool addLostHits );
  
   TLegend             *m_legend;
};
