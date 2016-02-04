/*
 *  FWL1EtMissParticleGlimpseProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

class FWL1EtMissParticleGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleGlimpseProxyBuilder( void ) {}
   virtual ~FWL1EtMissParticleGlimpseProxyBuilder( void ) {}
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWL1EtMissParticleGlimpseProxyBuilder( const FWL1EtMissParticleGlimpseProxyBuilder& );    // stop default
   const FWL1EtMissParticleGlimpseProxyBuilder& operator=( const FWL1EtMissParticleGlimpseProxyBuilder& );    // stop default
   
   virtual void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWL1EtMissParticleGlimpseProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   fireworks::addDashedArrow( iData.phi(), iData.et(), &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1EtMissParticleGlimpseProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kGlimpseBit );

