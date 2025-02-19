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
// $Id: FWL1EtMissParticleProxyBuilder.cc,v 1.9 2010/09/16 15:42:20 yana Exp $
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

class FWL1EtMissParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleProxyBuilder( void ) {}
   virtual ~FWL1EtMissParticleProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EtMissParticleProxyBuilder( const FWL1EtMissParticleProxyBuilder& );    // stop default
   const FWL1EtMissParticleProxyBuilder& operator=( const FWL1EtMissParticleProxyBuilder& );    // stop default
  
   virtual void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWL1EtMissParticleProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   double scale = 10;

   fireworks::addDashedLine( iData.phi(), iData.theta(), iData.pt() * scale, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1EtMissParticleProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kAllRPZBits );
