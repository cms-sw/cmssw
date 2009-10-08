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
// $Id: FWElectronDetailView.h,v 1.15 2009/09/06 12:53:44 dmytro Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "TEveWindow.h"

class TEveCaloDataVec;
class TEveCaloLego;
class TGLViewer;

class FWElectronDetailView : public FWDetailView<reco::GsfElectron> {

public:
   FWElectronDetailView();
   virtual ~FWElectronDetailView();

   virtual void build (const FWModelId &id, const reco::GsfElectron*, TEveWindowSlot*);


   virtual void setBackgroundColor(Color_t col);

private:

   FWElectronDetailView(const FWElectronDetailView&); // stop default
   const FWElectronDetailView& operator=(const FWElectronDetailView&); // stop default

   virtual double deltaEtaSuperClusterTrackAtVtx (const reco::GsfElectron &);
   virtual double deltaPhiSuperClusterTrackAtVtx (const reco::GsfElectron &);
   virtual math::XYZPoint trackPositionAtCalo (const reco::GsfElectron &);
   virtual void addTrackPointsInCaloData(const reco::GsfElectron*, TEveCaloLego*);

   void drawCrossHair(const reco::GsfElectron*, TEveCaloLego*, TEveElementList*);
   void addInfo(const reco::GsfElectron*, TEveElementList*);
   void makeLegend (const reco::GsfElectron*, const FWModelId&, TCanvas*);
   Bool_t checkRange(Double_t &, Double_t&, Double_t &, Double_t&, Double_t, Double_t);

   TGLViewer* m_viewer;
};

#endif
