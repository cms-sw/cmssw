// -*- C++ -*-
#ifndef Fireworks_Electrons_FWMuonDetailView_h
#define Fireworks_Electrons_FWMuonDetailView_h

//
// Package:     Electrons
// Class  :     FWMuonDetailView
// $Id: FWMuonDetailView.h,v 1.5 2010/01/14 15:55:14 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"

class FWECALDetailViewBuilder;
namespace reco {
  class Muon;
}

class FWMuonDetailView : public FWDetailViewGL<reco::Muon> {

public:
   FWMuonDetailView();
   virtual ~FWMuonDetailView();


private:
   FWMuonDetailView(const FWMuonDetailView&); // stop default
   const FWMuonDetailView& operator=(const FWMuonDetailView&); // stop default

   virtual void build (const FWModelId &id, const reco::Muon*);
   virtual void setTextInfo(const FWModelId&, const reco::Muon*);

   void addSceneInfo(const reco::Muon *i, TEveElementList* tList);

   TEveCaloData* m_data;
   FWECALDetailViewBuilder* m_builder;
};

#endif
