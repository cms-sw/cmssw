// -*- C++ -*-
//
// Package:     TRacks
// Class  :     FWTrackDetailView
//
// Original Author:  Chad Jarvis
//         Created:  Wed Mar  7 09:13:47 EST 2008
//
// Implementation:
//      use following table pasted from HitPattern.h
//
//      +-----+-----+-----+-----+-----+-----+-----+-----+----------------+-----+-----+
//      |tk/mu|  sub-structure  |   sub-sub-structure   |     stereo     |  hit type |
//      +-----+-----+-----+-----+-----+-----+-----+-----+----------------+-----+-----+
//      | 10  |   9    8     7  |   6    5     4     3  |        2       |  1     0  | bit
//
//      |tk = 1      PXB = 1            layer = 1-3                       hit type = 0-3
//      |tk = 1      PXF = 2            disk  = 1-2                       hit type = 0-3
//      |tk = 1      TIB = 3            layer = 1-4      0=rphi,1=stereo  hit type = 0-3
//      |tk = 1      TID = 4            wheel = 1-3      0=rphi,1=stereo  hit type = 0-3
//      |tk = 1      TOB = 5            layer = 1-6      0=rphi,1=stereo  hit type = 0-3
//      |tk = 1      TEC = 6            wheel = 1-9      0=rphi,1=stereo  hit type = 0-3
//      |mu = 0      DT  = 1            layer                             hit type = 0-3
//      |mu = 0      CSC = 2            layer                             hit type = 0-3
//      |mu = 0      RPC = 3            layer                             hit type = 0-3
//      |mu = 0      GEM = 3            layer                             hit type = 0-3
//
//      hit type, see DataFormats/TrackingRecHit/interface/TrackingRecHit.h
//      valid    = valid hit                                     = 0
//      missing  = detector is good, but no rec hit found        = 1
//      inactive = detector is off, so there was no hope         = 2
//      bad      = there were many bad strips within the ellipse = 3
//

#include "Rtypes.h"
#include "Fireworks/Core/interface/FWDetailViewCanvas.h"

class FWGeometry;
class FWModelId;
class TEveWindowSlot;
class TEveWindow;

namespace reco {
   class Track;
}

class FWTrackResidualDetailView : public FWDetailViewCanvas<reco::Track>{
public:
   FWTrackResidualDetailView();
   virtual ~FWTrackResidualDetailView();

private:
   FWTrackResidualDetailView(const FWTrackResidualDetailView&); // stop default
   const FWTrackResidualDetailView& operator=(const FWTrackResidualDetailView&); // stop default

   virtual void build (const FWModelId &id, const reco::Track*);
   virtual void setTextInfo(const FWModelId &id, const reco::Track*);

   double getSignedResidual (const FWGeometry *geom, unsigned int id, double resX);
   void prepareData(const FWModelId &id, const reco::Track*);
   void printDebug();

   int m_ndet;
   int m_nhits;
   int m_det[64];
   float res[2][64];
   int hittype[64];
   int stereo[64];
   int substruct[64];
   int subsubstruct[64];
   int m_detector[64];

   Int_t   m_resXFill;
   Color_t m_resXCol;
   Int_t   m_resYFill;
   Color_t m_resYCol;
   Int_t   m_stereoFill;
   Color_t m_stereoCol;
   Int_t   m_invalidFill;
   Color_t m_invalidCol;

   const static char* m_det_tracker_str[];
};
