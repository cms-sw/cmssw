// -*- C++ -*-
#ifndef Fireworks_Electrons_FWElectronDetailView_h
#define Fireworks_Electrons_FWElectronDetailView_h

//
// Package:     Electrons
// Class  :     FWElectronDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWElectronDetailView.h,v 1.10 2009/08/22 17:10:22 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "TEveWindow.h"

class TEveCaloDataVec;
class TEveCaloLego;

class FWElectronDetailView : public FWDetailView<reco::GsfElectron> {

public:
   FWElectronDetailView();
   virtual ~FWElectronDetailView();

   virtual void build (const FWModelId &id, const reco::GsfElectron*, TEveWindowSlot*);

private:
   FWElectronDetailView(const FWElectronDetailView&); // stop default
   const FWElectronDetailView& operator=(const FWElectronDetailView&); // stop default
   void makeLegend(const reco::GsfElectron &electron, TCanvas* textCanvas);
   virtual double deltaEtaSuperClusterTrackAtVtx (const reco::GsfElectron &);
   virtual double deltaPhiSuperClusterTrackAtVtx (const reco::GsfElectron &);
   virtual math::XYZPoint trackPositionAtCalo (const reco::GsfElectron &);

   void  makeExtraLegend (const reco::GsfElectron*, TCanvas*);
   Bool_t checkRange(Double_t &, Double_t&, Double_t &, Double_t&, Double_t, Double_t);

   void drawCrossHair(const reco::GsfElectron*, int, TEveCaloLego*, TEveElementList*);

   Float_t m_unitCM;
};

#endif
