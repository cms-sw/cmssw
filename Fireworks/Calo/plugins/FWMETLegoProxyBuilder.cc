/*
 *  FWMETLegoProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/METReco/interface/MET.h"

class FWMETLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETLegoProxyBuilder( void ) {}
   ~FWMETLegoProxyBuilder( void ) override {}
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWMETLegoProxyBuilder( const FWMETLegoProxyBuilder& ) = delete;    // stop default
   const FWMETLegoProxyBuilder& operator=( const FWMETLegoProxyBuilder& ) = delete;    // stop default
   
   using FWSimpleProxyBuilderTemplate<reco::MET>::build;
   void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) override;
};

void
FWMETLegoProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   fireworks::addDoubleLines( iData.phi(), &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWMETLegoProxyBuilder, reco::MET, "recoMET", FWViewType::kAllLegoBits );
