// -*- C++ -*-
// $Id: FWSiStripClusterProxyBuilder.cc,v 1.5 2010/04/20 20:49:44 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiStripClusterProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSiStripClusterProxyBuilder() {}
  
   virtual ~FWSiStripClusterProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product );
  
   FWSiStripClusterProxyBuilder( const FWSiStripClusterProxyBuilder& );    // stop default
   const FWSiStripClusterProxyBuilder& operator=( const FWSiStripClusterProxyBuilder& );    // stop default
   void modelChanges( const FWModelIds& iIds, TEveElement* iElements );
   void applyChangesToAllModels( TEveElement* iElements );
};

void
FWSiStripClusterProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product )
{
   const edmNew::DetSetVector<SiStripCluster>* clusters = 0;
   iItem->get( clusters );
   if( 0 == clusters ) return;
   
   for( edmNew::DetSetVector<SiStripCluster>::const_iterator set = clusters->begin(), setEnd = clusters->end();
       set != setEnd; ++set) {
      unsigned int id = set->detId();
      TEveCompound* compound = createCompound();
      if( iItem->getGeom() ) {
	TEveGeoShape* shape = iItem->getGeom()->getShape( id );
	if( 0 != shape ) {
            shape->SetMainTransparency( 75 );
	    setupAddElement( shape, compound );
         }
      }
      TEveStraightLineSet *scposition = new TEveStraightLineSet( "strip" );
      for( edmNew::DetSet<SiStripCluster>::const_iterator ic = set->begin (), icEnd = set->end (); ic != icEnd; ++ic ) { 
	 double bc = (*ic).barycenter();
	 TVector3 point;
	 TVector3 pointA;
	 TVector3 pointB;
	 fireworks::localSiStrip( point, pointA, pointB, bc, id, iItem );
	 scposition->AddLine( pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z() );
	 scposition->SetLineColor( kRed );
      }
      product->AddElement( scposition );
      setupAddElement( compound, product );
   }
}

void
FWSiStripClusterProxyBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}
 
void
FWSiStripClusterProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   if(0!=iElements && item() && item()->size()) {
     //make the bad assumption that everything is being changed indentically
     const FWEventItem::ModelInfo info(item()->defaultDisplayProperties(),false);
   }
}

REGISTER_FWPROXYBUILDER( FWSiStripClusterProxyBuilder, edmNew::DetSetVector<SiStripCluster>, "SiStripCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
