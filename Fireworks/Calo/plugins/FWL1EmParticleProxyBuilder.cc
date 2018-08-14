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
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"

class FWL1EmParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EmParticle>
{
public:
   FWL1EmParticleProxyBuilder( void ) {}
   ~FWL1EmParticleProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EmParticleProxyBuilder( const FWL1EmParticleProxyBuilder& ) = delete;    // stop default
   const FWL1EmParticleProxyBuilder& operator=( const FWL1EmParticleProxyBuilder& ) = delete;    // stop default
  
   using FWSimpleProxyBuilderTemplate<l1extra::L1EmParticle>::build;
   void build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) override;
};

void
FWL1EmParticleProxyBuilder::build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   double scale = 10;

   fireworks::addDashedLine( iData.phi(), iData.theta(), iData.pt() * scale, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1EmParticleProxyBuilder, l1extra::L1EmParticle, "L1EmParticle", FWViewType::kAllRPZBits );
