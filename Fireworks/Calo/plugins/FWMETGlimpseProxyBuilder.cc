/*
 *  FWMETGlimpseProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Calo/interface/CaloUtils.h"

#include "DataFormats/METReco/interface/MET.h"

class FWMETGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETGlimpseProxyBuilder( void ) {}
   virtual ~FWMETGlimpseProxyBuilder( void ) {}
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWMETGlimpseProxyBuilder( const FWMETGlimpseProxyBuilder& );    // stop default
   const FWMETGlimpseProxyBuilder& operator=( const FWMETGlimpseProxyBuilder& );    // stop default
   
   virtual void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* );
};

void 
FWMETGlimpseProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* ) 
{
   fireworks::addDashedArrow( iData.phi(), iData.et(), &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWMETGlimpseProxyBuilder, reco::MET, "recoMET", FWViewType::kGlimpseBit );
