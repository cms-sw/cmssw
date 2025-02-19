// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1EmParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1EmParticleProxyBuilder.cc,v 1.7 2010/09/16 15:42:20 yana Exp $
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"

class FWL1EmParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EmParticle>
{
public:
   FWL1EmParticleProxyBuilder( void ) {}
   virtual ~FWL1EmParticleProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EmParticleProxyBuilder( const FWL1EmParticleProxyBuilder& );    // stop default
   const FWL1EmParticleProxyBuilder& operator=( const FWL1EmParticleProxyBuilder& );    // stop default
  
   virtual void build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* );
};

void
FWL1EmParticleProxyBuilder::build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   double scale = 10;

   fireworks::addDashedLine( iData.phi(), iData.theta(), iData.pt() * scale, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1EmParticleProxyBuilder, l1extra::L1EmParticle, "L1EmParticle", FWViewType::kAllRPZBits );
