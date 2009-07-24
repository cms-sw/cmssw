// -*- C++ -*-
#ifndef Fireworks_Electrons_FWECALDetailView_h
#define Fireworks_Electrons_FWECALDetailView_h

//
// Package:     Electrons
// Class  :     FWECALDetailView
//
// Implementation: Common base class for electron and photon detail
//     views.  The only difference between the two detail views is
//     whether the track intersections need to be drawn.
//
// $Id: FWECALDetailView.h,v 1.3 2009/04/27 16:53:30 dmytro Exp $
//

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "Fireworks/Core/interface/FWDetailView.h"

class TEveElement;
class TEveElementList;
class TEveCaloDataVec;
class FWModelId;
class TGLViewerBase;
class TGLOverlayElement;

template <typename T> class FWECALDetailView : public FWDetailView<T> {
public:
     FWECALDetailView ();
     virtual ~FWECALDetailView ();

    virtual  void  clearOverlayElements();

protected:
     void setItem (const FWEventItem *iItem) {
	  m_item = iItem;
     }
     virtual bool	drawTrack () = 0;
     virtual math::XYZPoint trackPositionAtCalo (const T &);
     virtual double deltaEtaSuperClusterTrackAtVtx (const T &);
     virtual double deltaPhiSuperClusterTrackAtVtx (const T &);
     void build_projected (const FWModelId &id, const T *, 
			   TEveElementList *);
     virtual class TEveElementList *makeLabels (const T &) = 0;
     
     void fillData (const std::vector< std::pair<DetId, float> > &detids,
		    TEveCaloDataVec *data, 
		    double phi_seed);
     void getEcalCrystalsEndcap (std::vector<DetId> *, 
				 int ix, int iy, int iz);
     void getEcalCrystalsBarrel (std::vector<DetId> *, 
				 int ieta, int iphi);
    
     std::vector<TGLOverlayElement*> m_overlays;
protected:
   // ---------- member data --------------------------------
     const FWEventItem* m_item;
     
     bool  m_coordEtaPhi; // use XY coordinate if EndCap, else EtaPhi
     
     const EcalRecHitCollection *m_barrel_hits;
     const EcalRecHitCollection *m_endcap_hits;
     const EcalRecHitCollection *m_endcap_reduced_hits;
     std::vector< std::pair<DetId, float> > seed_detids;
};

#include "FWECALDetailView.icc"

#endif
