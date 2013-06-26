// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1MuonParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1MuonParticleProxyBuilder.cc,v 1.10 2010/09/16 15:42:20 yana Exp $
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

class FWL1MuonParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1MuonParticle>
{
public:
   FWL1MuonParticleProxyBuilder( void ) {}
   virtual ~FWL1MuonParticleProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1MuonParticleProxyBuilder( const FWL1MuonParticleProxyBuilder& );    // stop default
   const FWL1MuonParticleProxyBuilder& operator=( const FWL1MuonParticleProxyBuilder& );    // stop default
  
   virtual void build( const l1extra::L1MuonParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* );
};

void
FWL1MuonParticleProxyBuilder::build( const l1extra::L1MuonParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   double scale = 10;

   fireworks::addDashedLine( iData.phi(), iData.theta(), iData.pt() * scale, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER(FWL1MuonParticleProxyBuilder, l1extra::L1MuonParticle, "L1MuonParticle", FWViewType::kAllRPZBits);
