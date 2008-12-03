// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonsLegoProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 17:40:15 EST 2008
// $Id$
//

// system include files
#include "TEvePointSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

class FWMuonsLegoProxyBuilder : public FW3DLegoSimpleProxyBuilderTemplate<reco::Muon> {
   
public:
   FWMuonsLegoProxyBuilder();
   //virtual ~FWMuonsLegoProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
  
private:
   FWMuonsLegoProxyBuilder(const FWMuonsLegoProxyBuilder&); // stop default
   
   const FWMuonsLegoProxyBuilder& operator=(const FWMuonsLegoProxyBuilder&); // stop default
   
   void build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   // ---------- member data --------------------------------
      
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWMuonsLegoProxyBuilder::FWMuonsLegoProxyBuilder()
{
}

// FWMuonsLegoProxyBuilder::FWMuonsLegoProxyBuilder(const FWMuonsLegoProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

//FWMuonsLegoProxyBuilder::~FWMuonsLegoProxyBuilder()
//{
//}

//
// assignment operators
//
// const FWMuonsLegoProxyBuilder& FWMuonsLegoProxyBuilder::operator=(const FWMuonsLegoProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWMuonsLegoProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
void 
FWMuonsLegoProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   TEvePointSet* points = new TEvePointSet("points");
   oItemHolder.AddElement(points);
   
   points->SetMarkerStyle(2);
   points->SetMarkerSize(0.2);
   if ( iData.track().isAvailable() && iData.track()->extra().isAvailable() ) {
      // get position of the muon at surface of the tracker
      points->SetNextPoint(iData.track()->outerPosition().eta(),
                           iData.track()->outerPosition().phi(),
                           0.1);
   } else {
      if ( iData.standAloneMuon().isAvailable() && iData.standAloneMuon()->extra().isAvailable() ) {
         // get position of the inner state of the stand alone muon
         if (  iData.standAloneMuon()->innerPosition().R() <  iData.standAloneMuon()->outerPosition().R() )
            points->SetNextPoint(iData.standAloneMuon()->innerPosition().eta(),
                                 iData.standAloneMuon()->innerPosition().phi(),
                                 0.1);
         else
            points->SetNextPoint(iData.standAloneMuon()->outerPosition().eta(),
                                 iData.standAloneMuon()->outerPosition().phi(),
                                 0.1);
      } else {
         // WARNING: use direction at POCA as the last option
         points->SetNextPoint(iData.eta(),iData.phi(),0.1);
      }
   }
   points->SetMarkerColor(  item()->defaultDisplayProperties().color() );
   
}

//
// static member functions
//

REGISTER_FW3DLEGODATAPROXYBUILDER(FWMuonsLegoProxyBuilder,reco::Muon,"Muons");
