// -*- C++ -*-
// $Id: FWSiStripClusterProxyBuilder.cc,v 1.14 2010/08/19 13:39:17 yana Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class FWSiStripClusterProxyBuilder : public FWSimpleProxyBuilderTemplate<SiStripCluster>
{
public:
   FWSiStripClusterProxyBuilder( void ) {}
   virtual ~FWSiStripClusterProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

protected:
   virtual void build( const SiStripCluster& iData, unsigned int iIndex,
		       TEveElement& oItemHolder, const FWViewContext* );
   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
				   FWViewType::EType viewType, const FWViewContext* vc );

private:
   FWSiStripClusterProxyBuilder( const FWSiStripClusterProxyBuilder& );
   const FWSiStripClusterProxyBuilder& operator=( const FWSiStripClusterProxyBuilder& );              
};

void
FWSiStripClusterProxyBuilder::build( const SiStripCluster& iData,           
				     unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  unsigned int rawid = iData.geographicalId();
  const DetIdToMatrix *geom = item()->getGeom();
 
  TEveGeoShape* shape = geom->getShape( rawid );
  
  if( shape ) 
  {
    shape->SetElementName( "Det" );
    setupAddElement( shape, &oItemHolder );
  }
  else
  {
    fwLog( fwlog::kWarning ) 
      << "failed to get shape of SiStripCluster with detid: "
      << rawid << std::endl;
  }
  increaseComponentTransparency( iIndex, &oItemHolder, "Det", 60 );
  
  TEveStraightLineSet *lineSet = new TEveStraightLineSet( "strip" );
  setupAddElement( lineSet, &oItemHolder );

  const TGeoHMatrix* matrix = geom->getMatrix( rawid );
  const float* pars = geom->getParameters( rawid );
  if( pars == 0 || (! matrix ))
  {
    fwLog( fwlog::kError )
      << "failed to get topology of SiStripCluster with detid: " 
      << rawid << std::endl;

      return;
  }

  short firststrip = iData.firstStrip();
  double localTop[3] = { 0.0, 0.0, 0.0 };
  double localBottom[3] = { 0.0, 0.0, 0.0 };

  fireworks::localSiStrip( firststrip, localTop, localBottom, pars, rawid );

  double globalTop[3];
  double globalBottom[3];
  matrix->LocalToMaster( localTop, globalTop );
  matrix->LocalToMaster( localBottom, globalBottom );
  
  lineSet->AddLine( globalTop[0], globalTop[1], globalTop[2],
		    globalBottom[0], globalBottom[1], globalBottom[2] );  
}

void
FWSiStripClusterProxyBuilder::localModelChanges( const FWModelId& iId, TEveElement* iCompound,
						 FWViewType::EType viewType, const FWViewContext* vc )
{
  increaseComponentTransparency( iId.index(), iCompound, "Det", 60 );
}


REGISTER_FWPROXYBUILDER( FWSiStripClusterProxyBuilder, SiStripCluster, "SiStripCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
