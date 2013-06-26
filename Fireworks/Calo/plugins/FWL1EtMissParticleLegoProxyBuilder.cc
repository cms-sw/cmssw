/*
 *  FWL1EtMissParticleLegoProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

class FWL1EtMissParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleLegoProxyBuilder( void ) {}
   virtual ~FWL1EtMissParticleLegoProxyBuilder( void ) {}
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWL1EtMissParticleLegoProxyBuilder( const FWL1EtMissParticleLegoProxyBuilder& );    // stop default
   const FWL1EtMissParticleLegoProxyBuilder& operator=( const FWL1EtMissParticleLegoProxyBuilder& );    // stop default
   
   virtual void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* );
};

void
FWL1EtMissParticleLegoProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   fireworks::addDoubleLines( iData.phi(), &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1EtMissParticleLegoProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kAllLegoBits );
