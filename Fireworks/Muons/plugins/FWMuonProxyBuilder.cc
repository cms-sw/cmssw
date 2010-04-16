// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  4 19:28:07 EST 2008
// $Id: FWMuonProxyBuilder.cc,v 1.2 2010/04/14 15:53:26 yana Exp $
//

#include "TEvePointSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWMuonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonProxyBuilder() {}
   virtual ~FWMuonProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonProxyBuilder(const FWMuonProxyBuilder&); // stop default

   const FWMuonProxyBuilder& operator=(const FWMuonProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   m_builder.buildMuon(item(), &iData, &oItemHolder, true, false);
   
   // FIXME: To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   // m_builder.buildMuon(item(), &iData, &oItemHolder, false, false);
}

class FWMuonRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonRhoPhiProxyBuilder() {}
   virtual ~FWMuonRhoPhiProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonRhoPhiProxyBuilder(const FWMuonRhoPhiProxyBuilder&); // stop default

   const FWMuonRhoPhiProxyBuilder& operator=(const FWMuonRhoPhiProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonRhoPhiProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   // To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   m_builder.buildMuon(item(), &iData, &oItemHolder, false, false);
}

class FWMuonLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonLegoProxyBuilder() {}
   virtual ~FWMuonLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonLegoProxyBuilder(const FWMuonLegoProxyBuilder&); // stop default
   const FWMuonLegoProxyBuilder& operator=(const FWMuonLegoProxyBuilder&); // stop default

   virtual void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonLegoProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   TEvePointSet* points = new TEvePointSet("points");
   oItemHolder.AddElement(points);
 
   points->SetMarkerStyle(2);
   points->SetMarkerSize(0.2);
    
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

class FWMuonGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonGlimpseProxyBuilder() {}
   virtual ~FWMuonGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonGlimpseProxyBuilder(const FWMuonGlimpseProxyBuilder&); // stop default

   const FWMuonGlimpseProxyBuilder& operator=(const FWMuonGlimpseProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWMuonGlimpseProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder) const 
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet( "", "");
   marker->SetLineWidth(2);
   fireworks::addStraightLineSegment( marker, &iData, 1.0 );
   oItemHolder.AddElement(marker);
   //add to scaler at end so that it can scale the line after all ends have been added
   // FIXME:   scaler()->addElement(marker);
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWMuonProxyBuilder, reco::Muon, "Muons", FWViewType::k3DBit | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWMuonRhoPhiProxyBuilder, reco::Muon, "Muons", FWViewType::kRhoPhiBit);
REGISTER_FWPROXYBUILDER(FWMuonLegoProxyBuilder, reco::Muon, "Muons", FWViewType::kLegoBit);
REGISTER_FWPROXYBUILDER(FWMuonGlimpseProxyBuilder, reco::Muon, "Muons", FWViewType::kGlimpseBit);
