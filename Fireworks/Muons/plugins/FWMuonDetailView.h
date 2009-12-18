// -*- C++ -*-
#ifndef Fireworks_Electrons_FWMuonDetailView_h
#define Fireworks_Electrons_FWMuonDetailView_h

//
// Package:     Electrons
// Class  :     FWMuonDetailView
// $Id: FWMuonDetailView.h,v 1.3 2009/10/27 19:25:03 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuonDetailView : public FWDetailView<reco::Muon> {

public:
   FWMuonDetailView();
   virtual ~FWMuonDetailView();

   virtual void build (const FWModelId &id, const reco::Muon*, TEveWindowSlot*);

   virtual void setBackgroundColor(Color_t col);
private:
   FWMuonDetailView(const FWMuonDetailView&); // stop default
   const FWMuonDetailView& operator=(const FWMuonDetailView&); // stop default

   double makeLegend(double x0, 
		     double y0, 
		     const reco::Muon *muon, 
		     const FWModelId& id);
   void addInfo(const reco::Muon *i, TEveElementList* tList);

   TGLViewer*    m_viewer;
   TEveCaloData* m_data;
};

#endif
