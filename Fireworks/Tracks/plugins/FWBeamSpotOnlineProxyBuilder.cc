/*
 *  FWBeamSpotProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 7/29/10.
 *
 */

#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/Scalers/interface/BeamSpotOnline.h"

class FWBeamSpotOnlineProxyBuilder : public FWSimpleProxyBuilderTemplate<BeamSpotOnline>
{
public:
  FWBeamSpotOnlineProxyBuilder( void ) {}
  ~FWBeamSpotOnlineProxyBuilder( void ) override {}
   
  REGISTER_PROXYBUILDER_METHODS();

private:
  // Disable default copy constructor
  FWBeamSpotOnlineProxyBuilder( const FWBeamSpotOnlineProxyBuilder& ) = delete;
  // Disable default assignment operator
  const FWBeamSpotOnlineProxyBuilder& operator=( const FWBeamSpotOnlineProxyBuilder& ) = delete;

  using FWSimpleProxyBuilderTemplate<BeamSpotOnline>::build;
  void build( const BeamSpotOnline& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWBeamSpotOnlineProxyBuilder::build( const BeamSpotOnline& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  TEvePointSet* pointSet = new TEvePointSet;
  setupAddElement( pointSet, &oItemHolder );

  TEveStraightLineSet* lineSet = new TEveStraightLineSet;
  setupAddElement( lineSet, &oItemHolder );

  double posx = iData.x();
  double posy = iData.y();
  double posz = iData.z();
  double errx = iData.err_x();
  double erry = iData.err_y();
  double errz = iData.err_z();
  
  pointSet->SetNextPoint( posx, posy, posz ); 
  pointSet->SetNextPoint( posx + errx, posy + erry, posz + errz );
  pointSet->SetNextPoint( posx - errx, posy - erry, posz - errz );
  
  lineSet->AddLine( posx + errx, posy + erry, posz + errz,
		    posx - errx, posy - erry, posz - errz );
}

REGISTER_FWPROXYBUILDER( FWBeamSpotOnlineProxyBuilder, BeamSpotOnline, "Beam Spot Online",  FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
