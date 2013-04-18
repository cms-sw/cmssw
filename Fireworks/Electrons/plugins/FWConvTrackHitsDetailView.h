// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWConvTrackHitsDetailView

#include "Rtypes.h"

#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class TGLEmbeddedViewer;
class FWIntValueListener;
namespace reco {
   class Conversion;
}

class FWConvTrackHitsDetailView: public FWDetailViewGL<reco::Conversion>,
                             public CSGActionSupervisor
{
public:
   FWConvTrackHitsDetailView();
   virtual ~FWConvTrackHitsDetailView();

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
   FWConvTrackHitsDetailView(const FWConvTrackHitsDetailView&); // stop default
   const FWConvTrackHitsDetailView& operator=(const FWConvTrackHitsDetailView&); // stop default

   void build (const FWModelId &id, const reco::Conversion*);
   void setTextInfo (const FWModelId &id, const reco::Conversion*);
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
