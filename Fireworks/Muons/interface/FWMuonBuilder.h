#ifndef Fireworks_Muons_FWMuonBuilder_h
#define Fireworks_Muons_FWMuonBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
//
// $Id: FWMuonBuilder.h,v 1.3 2009/01/23 21:35:46 amraktad Exp $
//
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
namespace reco
{
   class Muon;
   class TrackExtra;
}
namespace fw
{
   class NamedCounter;
}

class FWEventItem;
class TEveElementList;
class TEveTrackPropagator;
class CmsMagField;

class FWMuonBuilder
{

public:
   FWMuonBuilder();
   virtual ~FWMuonBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void buildMuon(const FWEventItem* iItem,
                  const reco::Muon* muon,
                  TEveElement* tList,
                  bool showEndcap,
                  bool onlyTracks = false);


private:
   FWMuonBuilder(const FWMuonBuilder&);    // stop default

   const FWMuonBuilder& operator=(const FWMuonBuilder&);    // stop default

   void calculateField(const reco::Muon& iData);

   // ---------- member data --------------------------------
   FWEvePtr<TEveTrackPropagator> m_propagator;
   CmsMagField* m_cmsMagField;
};


#endif
