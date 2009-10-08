// -*- C++ -*-
#ifndef Fireworks_Electrons_FWPhotonDetailView_h
#define Fireworks_Electrons_FWPhotonDetailView_h
//
// Package:     Calo
// Class  :     FWPhotonDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWPhotonDetailView.h,v 1.5 2009/09/06 12:54:23 dmytro Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "TEveWindow.h"

class TEveWindowSlot;
class TGLViewer;

class FWPhotonDetailView : public FWDetailView<reco::Photon> {

public:
   FWPhotonDetailView();
   virtual ~FWPhotonDetailView();

   virtual void build (const FWModelId &id, const reco::Photon*, TEveWindowSlot*);

   virtual void setBackgroundColor(Color_t col);

private:
   FWPhotonDetailView(const FWPhotonDetailView&); // stop default
   const FWPhotonDetailView& operator=(const FWPhotonDetailView&); // stop default
   void makeLegend(const reco::Photon*, const FWModelId&, TCanvas*);
   void addInfo(const reco::Photon*, TEveElementList*);

   TGLViewer* m_viewer;
};

#endif
