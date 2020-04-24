// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1EtMissParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

class FWL1EtMissParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleProxyBuilder( void ) {}
   ~FWL1EtMissParticleProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EtMissParticleProxyBuilder( const FWL1EtMissParticleProxyBuilder& ) = delete;    // stop default
   const FWL1EtMissParticleProxyBuilder& operator=( const FWL1EtMissParticleProxyBuilder& ) = delete;    // stop default
  
   using FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>::build;
   void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWL1EtMissParticleProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   double scale = 10;

   fireworks::addDashedLine( iData.phi(), iData.theta(), iData.pt() * scale, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1EtMissParticleProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kAllRPZBits );
