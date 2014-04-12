/*
 *  FWL1MuonParticleLegoProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

class FWL1MuonParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1MuonParticle>
{
public:
   FWL1MuonParticleLegoProxyBuilder( void ) {}
   virtual ~FWL1MuonParticleLegoProxyBuilder( void ) {}
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWL1MuonParticleLegoProxyBuilder( const FWL1MuonParticleLegoProxyBuilder& );    // stop default
   const FWL1MuonParticleLegoProxyBuilder& operator=( const FWL1MuonParticleLegoProxyBuilder& );    // stop default
   
   virtual void build( const l1extra::L1MuonParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* );
};

void
FWL1MuonParticleLegoProxyBuilder::build( const l1extra::L1MuonParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   fireworks::addCircle( iData.eta(), iData.phi(), 0.5, 10, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWL1MuonParticleLegoProxyBuilder, l1extra::L1MuonParticle, "L1MuonParticle", FWViewType::kAllLegoBits );
