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
// $Id: FWECALDetailView.h,v 1.8 2009/08/22 17:10:22 amraktad Exp $
//
#include "Rtypes.h"

#include "Fireworks/Core/interface/FWDetailView.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/Point3D.h"


//class TGLViewerBase;
class TGLOverlayElement;
class TCanvas;
class TEveElement;
class TEveScene;
class TEveElementList;
class TEveCaloDataVec;
class TEveCaloLego;
class TEveCaloLegoOverlay;

class FWModelId;


template <typename T> class FWECALDetailView : public FWDetailView<T> {
public:
   FWECALDetailView ();
   virtual ~FWECALDetailView ();

protected:
   void setItem (const FWEventItem *iItem) {
      m_item = iItem;
   }
   virtual bool	drawTrack () = 0;
   virtual math::XYZPoint trackPositionAtCalo (const T &);
   virtual double deltaEtaSuperClusterTrackAtVtx (const T &);
   virtual double deltaPhiSuperClusterTrackAtVtx (const T &);
   virtual void build_projected (const FWModelId &id, const T *, TEveWindowSlot*);
   virtual void makeLegend(const T &, TCanvas*) = 0;

   void fillData (const std::vector< std::pair<DetId, float> > &detids,
                  TEveCaloDataVec *data,
                  double phi_seed);
   void getEcalCrystalsEndcap (std::vector<DetId> *,
                               int ix, int iy, int iz);
   void getEcalCrystalsBarrel (std::vector<DetId> *,
                               int ieta, int iphi);

   virtual void drawCrossHair(const T*, int, TEveCaloLego*, TEveElementList*) {}

   virtual void addTrackPointsInCaloData(const T*, int, TEveCaloDataVec*) {}

   Bool_t checkRange(Double_t &, Double_t&, Double_t &, Double_t&, Double_t, Double_t);

protected:
   virtual void build_projectedJohannes (const FWModelId &id, const T *, TEveWindowSlot*);
   virtual void build_projectedLothar (const FWModelId &id, const T *, TEveWindowSlot*);
   virtual void build_projectedDave (const FWModelId &id, const T *, TEveWindowSlot*);

   // ---------- member data --------------------------------
   Double_t   m_unitCM;
   const FWEventItem* m_item;

   bool  m_coordEtaPhi; // use XY coordinate if EndCap, else EtaPhi

   const EcalRecHitCollection *m_barrel_hits;
   const EcalRecHitCollection *m_endcap_hits;
   const EcalRecHitCollection *m_endcap_reduced_hits;
   std::vector< std::pair<DetId, float> > seed_detids;

};

#include "FWECALDetailView.icc"

#endif
