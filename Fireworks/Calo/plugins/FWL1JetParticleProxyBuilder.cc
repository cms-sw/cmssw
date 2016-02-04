// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1JetParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1JetParticleProxyBuilder.cc,v 1.9 2010/09/16 15:42:20 yana Exp $
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

class FWL1JetParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1JetParticle>
{
public:
   FWL1JetParticleProxyBuilder( void ) {}
   virtual ~FWL1JetParticleProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1JetParticleProxyBuilder( const FWL1JetParticleProxyBuilder& );    // stop default
   const FWL1JetParticleProxyBuilder& operator=( const FWL1JetParticleProxyBuilder& );    // stop default
  
   virtual void build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* );
};

void
FWL1JetParticleProxyBuilder::build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   double scale = 10;

   fireworks::addDashedLine( iData.phi(), iData.theta(), iData.pt() * scale, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1JetParticleProxyBuilder, l1extra::L1JetParticle, "L1JetParticle", FWViewType::kAllRPZBits );
