// -*- C++ -*-
// $Id: FWSiStripClusterProxyBuilder.cc,v 1.8 2010/05/03 10:36:42 mccauley Exp $
//

#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class FWSiStripClusterProxyBuilder : public FWSimpleProxyBuilderTemplate<SiStripCluster>
{
public:
   FWSiStripClusterProxyBuilder() {}
   virtual ~FWSiStripClusterProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWSiStripClusterProxyBuilder(const FWSiStripClusterProxyBuilder&);
   const FWSiStripClusterProxyBuilder& operator=(const FWSiStripClusterProxyBuilder&);              
  
   void build(const SiStripCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void
FWSiStripClusterProxyBuilder::build(const SiStripCluster& iData,           
                                    unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{  
  DetId detid(iData.geographicalId());
 
  TEveGeoShape* shape = item()->getGeom()->getShape(detid);
  
  if ( shape ) 
  {
    shape->SetMainTransparency(75);  // FIXME: Magic number
    setupAddElement(shape, &oItemHolder);
  }


  TEveStraightLineSet *scposition = new TEveStraightLineSet( "strip" );
  
  double bc = iData.barycenter();

  TVector3 point;
  TVector3 pointA;
  TVector3 pointB;

  fireworks::localSiStrip(point, pointA, pointB, bc, detid, item());

  scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), 
                      pointB.X(), pointB.Y(), pointB.Z() );
  
  scposition->SetLineColor(kRed);

  setupAddElement(scposition, &oItemHolder);
}


REGISTER_FWPROXYBUILDER( FWSiStripClusterProxyBuilder, SiStripCluster, "SiStripCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
