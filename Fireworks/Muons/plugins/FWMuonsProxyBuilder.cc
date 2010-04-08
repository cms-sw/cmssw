// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonsProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  4 19:28:07 EST 2008
// $Id: FWMuonsProxyBuilder.cc,v 1.3 2010/04/08 13:09:33 yana Exp $
//

#include "TEvePointSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWMuonsProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonsProxyBuilder() {}
   virtual ~FWMuonsProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonsProxyBuilder(const FWMuonsProxyBuilder&); // stop default

   const FWMuonsProxyBuilder& operator=(const FWMuonsProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonsProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   m_builder.buildMuon(item(), &iData, &oItemHolder, true, false);
   
   // FIXME: To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   // m_builder.buildMuon(item(), &iData, &oItemHolder, false, false);
}

class FWMuonsRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonsRhoPhiProxyBuilder() {}
   virtual ~FWMuonsRhoPhiProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonsRhoPhiProxyBuilder(const FWMuonsRhoPhiProxyBuilder&); // stop default

   const FWMuonsRhoPhiProxyBuilder& operator=(const FWMuonsRhoPhiProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonsRhoPhiProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   // To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   m_builder.buildMuon(item(), &iData, &oItemHolder, false, false);
}

class FWMuonsLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonsLegoProxyBuilder() {}
   virtual ~FWMuonsLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonsLegoProxyBuilder(const FWMuonsLegoProxyBuilder&); // stop default

   const FWMuonsLegoProxyBuilder& operator=(const FWMuonsLegoProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonsLegoProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   TEvePointSet* points = new TEvePointSet("points");
   oItemHolder.AddElement(points);
 
   points->SetMarkerStyle(2);
   points->SetMarkerSize(0.2);
   points->SetMarkerColor(  item()->defaultDisplayProperties().color() );
    
   // get ECAL position of the propagated trajectory if available
   if( iData.isEnergyValid() && iData.calEnergy().ecal_position.r()>100 ) {
      points->SetNextPoint(iData.calEnergy().ecal_position.eta(),
			   iData.calEnergy().ecal_position.phi(),
			   0.1);
     return;
   }
    
   // get position of the muon at surface of the tracker
   if( iData.track().isAvailable() && iData.track()->extra().isAvailable() ) {
      points->SetNextPoint(iData.track()->outerPosition().eta(),
			   iData.track()->outerPosition().phi(),
			   0.1);
      return;
   } 

   // get position of the inner state of the stand alone muon
   if( iData.standAloneMuon().isAvailable() && iData.standAloneMuon()->extra().isAvailable() ) {
      if( iData.standAloneMuon()->innerPosition().R() <  iData.standAloneMuon()->outerPosition().R() )
         points->SetNextPoint(iData.standAloneMuon()->innerPosition().eta(),
			      iData.standAloneMuon()->innerPosition().phi(),
			      0.1);
      else
         points->SetNextPoint(iData.standAloneMuon()->outerPosition().eta(),
			      iData.standAloneMuon()->outerPosition().phi(),
			      0.1);
      return;
   } 
   
   // WARNING: use direction at POCA as the last option
   points->SetNextPoint( iData.eta(), iData.phi(), 0.1 );
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWMuonsProxyBuilder, reco::Muon, "Muons", FWViewType::k3DBit | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWMuonsRhoPhiProxyBuilder, reco::Muon, "Muons", FWViewType::kRhoPhiBit);
REGISTER_FWPROXYBUILDER(FWMuonsLegoProxyBuilder, reco::Muon, "Muons", FWViewType::kLegoBit);
