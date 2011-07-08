/*
 *  FWBeamSpotProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 7/29/10.
 *
 */
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class FWBeamSpotProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::BeamSpot>
{
public:
  FWBeamSpotProxyBuilder( void ) {}
  virtual ~FWBeamSpotProxyBuilder( void ) {}
   
  REGISTER_PROXYBUILDER_METHODS();

private:
  // Disable default copy constructor
  FWBeamSpotProxyBuilder( const FWBeamSpotProxyBuilder& );
  // Disable default assignment operator
  const FWBeamSpotProxyBuilder& operator=( const FWBeamSpotProxyBuilder& );

  virtual void build( const reco::BeamSpot& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWBeamSpotProxyBuilder::build( const reco::BeamSpot& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  const reco::BeamSpot::Point &pos =  iData.position();
  TEvePointSet* pointSet = new TEvePointSet;
  pointSet->SetMarkerSize(0.5);
  setupAddElement( pointSet, &oItemHolder );

  pointSet->SetNextPoint( pos.x(),  pos.y(),  pos.z()); 
}

REGISTER_FWPROXYBUILDER( FWBeamSpotProxyBuilder, reco::BeamSpot, "Beam Spot",  FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
