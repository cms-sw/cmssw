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
// $Id: FWMuonProxyBuilder.cc,v 1.8 2010/05/21 13:45:46 mccauley Exp $
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
   virtual void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);

   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

   void setChamberTransparency(unsigned int index, TEveElement* holder);

   mutable FWMuonBuilder m_builder;
};

void
FWMuonProxyBuilder::setChamberTransparency(unsigned int index, TEveElement* holder)
{
   const FWDisplayProperties& dp = item()->modelInfo(index).displayProperties();
   Char_t chamber_transp = TMath::Min(100, 60 + dp.transparency() / 2);
   TEveElement::List_t chambers;
   holder->FindChildren(chambers, "Chamber");
   for (TEveElement::List_i c = chambers.begin(); c != chambers.end(); ++c)
   {
      (*c)->SetMainTransparency(chamber_transp);
   }
}

void
FWMuonProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   m_builder.buildMuon(this, &iData, &oItemHolder, true, false);

   setChamberTransparency(iIndex, &oItemHolder);

   // FIXME: To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   // m_builder.buildMuon(item(), &iData, &oItemHolder, false, false);
}

void
FWMuonProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                      FWViewType::EType viewType, const FWViewContext* vc)
{
   setChamberTransparency(iId.index(), iCompound);
}

//______________________________________________________________________________


class FWMuonRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonRhoPhiProxyBuilder() {}
   virtual ~FWMuonRhoPhiProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonRhoPhiProxyBuilder(const FWMuonRhoPhiProxyBuilder&); // stop default

   const FWMuonRhoPhiProxyBuilder& operator=(const FWMuonRhoPhiProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);

   mutable FWMuonBuilder m_builder;
};

void
FWMuonRhoPhiProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   // To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   m_builder.buildMuon(this, &iData, &oItemHolder, false, false);
}

//______________________________________________________________________________


class FWMuonLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonLegoProxyBuilder() {}
   virtual ~FWMuonLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonLegoProxyBuilder(const FWMuonLegoProxyBuilder&); // stop default
   const FWMuonLegoProxyBuilder& operator=(const FWMuonLegoProxyBuilder&); // stop default

   virtual void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void
FWMuonLegoProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   TEvePointSet* points = new TEvePointSet("points");
   setupAddElement(points, &oItemHolder);
 
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

//______________________________________________________________________________


class FWMuonGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonGlimpseProxyBuilder() {}
   virtual ~FWMuonGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonGlimpseProxyBuilder(const FWMuonGlimpseProxyBuilder&); // stop default

   const FWMuonGlimpseProxyBuilder& operator=(const FWMuonGlimpseProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void
FWMuonGlimpseProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet( "", "");
   marker->SetLineWidth(2);
   fireworks::addStraightLineSegment( marker, &iData, 1.0 );
   setupAddElement(marker, &oItemHolder);
   //add to scaler at end so that it can scale the line after all ends have been added
   // FIXME:   scaler()->addElement(marker);
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWMuonProxyBuilder, reco::Muon, "Muons", FWViewType::kAll3DBits | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWMuonRhoPhiProxyBuilder, reco::Muon, "Muons", FWViewType::kRhoPhiBit);
REGISTER_FWPROXYBUILDER(FWMuonLegoProxyBuilder, reco::Muon, "Muons", FWViewType::kLegoBit);
REGISTER_FWPROXYBUILDER(FWMuonGlimpseProxyBuilder, reco::Muon, "Muons", FWViewType::kGlimpseBit);
