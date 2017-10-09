/*
 *  FWMETGlimpseProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/METReco/interface/MET.h"

class FWMETGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETGlimpseProxyBuilder( void ) {}
   ~FWMETGlimpseProxyBuilder( void ) override {}
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWMETGlimpseProxyBuilder( const FWMETGlimpseProxyBuilder& ) = delete;    // stop default
   const FWMETGlimpseProxyBuilder& operator=( const FWMETGlimpseProxyBuilder& ) = delete;    // stop default
   
   using FWSimpleProxyBuilderTemplate<reco::MET>::build;
   void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) override;
};

void 
FWMETGlimpseProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   fireworks::addDashedArrow( iData.phi(), iData.et(), &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWMETGlimpseProxyBuilder, reco::MET, "recoMET", FWViewType::kGlimpseBit );
