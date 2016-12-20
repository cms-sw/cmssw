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
//
#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>
{
public:
   FWMuonProxyBuilder( void ) {}
   virtual ~FWMuonProxyBuilder( void ) {}

   virtual void setItem(const FWEventItem* iItem);

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWMuonProxyBuilder( const FWMuonProxyBuilder& );
   // Disable default assignment operator
   const FWMuonProxyBuilder& operator=( const FWMuonProxyBuilder& );


   using FWSimpleProxyBuilderTemplate<reco::Muon>::build;
   virtual void build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;

   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                   FWViewType::EType viewType, const FWViewContext* vc ) override;

   mutable FWMuonBuilder m_builder;
};


void
FWMuonProxyBuilder::setItem(const FWEventItem* iItem)
{
   FWProxyBuilderBase::setItem(iItem);
   
   if (iItem) {
      iItem->getConfig()->assertParam("LineWidth", long(1), long(1), long(4));
   }
}

void
FWMuonProxyBuilder::build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   int width = item()->getConfig()->value<long>("LineWidth");
   m_builder.setLineWidth(width);

   m_builder.buildMuon( this, &iData, &oItemHolder, true, false );
   increaseComponentTransparency( iIndex, &oItemHolder, "Chamber", 60 );
}

void
FWMuonProxyBuilder::localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                       FWViewType::EType viewType, const FWViewContext* vc )
{
   increaseComponentTransparency( iId.index(), iCompound, "Chamber", 60 );
}

REGISTER_FWPROXYBUILDER( FWMuonProxyBuilder, reco::Muon, "Muons", FWViewType::kAll3DBits | FWViewType::kRhoZBit );
