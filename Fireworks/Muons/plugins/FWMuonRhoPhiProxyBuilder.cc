// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonRhoPhiProxyBuilder
//
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuonRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>
{
public:
   FWMuonRhoPhiProxyBuilder( void ) {}
   virtual ~FWMuonRhoPhiProxyBuilder( void ) {}

   virtual void setItem(const FWEventItem* iItem);

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWMuonRhoPhiProxyBuilder( const FWMuonRhoPhiProxyBuilder& );
   // Disable default assignment operator
   const FWMuonRhoPhiProxyBuilder& operator=( const FWMuonRhoPhiProxyBuilder& );

   using FWSimpleProxyBuilderTemplate<reco::Muon>::build;
   void build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;

   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                   FWViewType::EType viewType, const FWViewContext* vc ) override;

   mutable FWMuonBuilder m_builder;
};

void
FWMuonRhoPhiProxyBuilder::setItem(const FWEventItem* iItem)
{
   FWProxyBuilderBase::setItem(iItem);
   
   if (iItem) {
      iItem->getConfig()->assertParam("LineWidth", long(1), long(1), long(4));
   }
}

void
FWMuonRhoPhiProxyBuilder::build( const reco::Muon& iData, unsigned int iIndex,
                                 TEveElement& oItemHolder, const FWViewContext* ) 
{
   int width = item()->getConfig()->value<long>("LineWidth");
   m_builder.setLineWidth(width);

   // To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   m_builder.buildMuon( this, &iData, &oItemHolder, false, false );

   increaseComponentTransparency( iIndex, &oItemHolder, "Chamber", 40 );
}

void
FWMuonRhoPhiProxyBuilder::localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                             FWViewType::EType viewType, const FWViewContext* vc )
{
   increaseComponentTransparency( iId.index(), iCompound, "Chamber", 40 );
}

REGISTER_FWPROXYBUILDER( FWMuonRhoPhiProxyBuilder, reco::Muon, "Muons", FWViewType::kRhoPhiBit |  FWViewType::kRhoPhiPFBit);
