#ifndef Fireworks_Muons_FWMuonBuilder_h
#define Fireworks_Muons_FWMuonBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
// 
/**\class FWMuonBuilder FWMuonBuilder.h Fireworks/Muons/interface/FWMuonBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 16:05:28 EST 2008
// $Id: FWMuonBuilder.h,v 1.1 2008/11/20 01:10:55 chrjones Exp $
//

// system include files

// user include files
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
      FWMuonBuilder(const FWMuonBuilder&); // stop default

      const FWMuonBuilder& operator=(const FWMuonBuilder&); // stop default

   void calculateField(const reco::Muon& iData);

      // ---------- member data --------------------------------
   FWEvePtr<TEveTrackPropagator> m_trackerPropagator;
   FWEvePtr<TEveTrackPropagator> m_innerPropagator;
   FWEvePtr<TEveTrackPropagator> m_outerPropagator;
   double m_magneticField;
   
};


#endif
