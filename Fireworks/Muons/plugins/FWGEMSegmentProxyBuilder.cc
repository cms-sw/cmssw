// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWGEMSegmentProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue Sep  29 12:36:00 EST 2015
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"

class FWGEMSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<GEMSegment>
{
public:
  FWGEMSegmentProxyBuilder( void ) {}
  virtual ~FWGEMSegmentProxyBuilder( void ) {}
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  FWGEMSegmentProxyBuilder( const FWGEMSegmentProxyBuilder& );   
  const FWGEMSegmentProxyBuilder& operator=( const FWGEMSegmentProxyBuilder& );

  using FWSimpleProxyBuilderTemplate<GEMSegment>::build;
  void build( const GEMSegment& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWGEMSegmentProxyBuilder::build( const GEMSegment& iData,           
				 unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  const FWGeometry *geom = item()->getGeom();
  unsigned int rawid = iData.gemDetId().rawId();
  
  if( ! geom->contains( rawid ))
  {
    fwLog(fwlog::kError) << "failed to get geometry of GEM chamber with rawid: " 
                         << rawid << std::endl;
    return;
  }
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet();
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 3 );
  setupAddElement( segmentSet, &oItemHolder );

  TEveGeoShape* shape = item()->getGeom()->getEveShape( rawid );
  // if( TGeoTrap* trap = dynamic_cast<TGeoTrap*>( shape->GetShape())) // Trapezoidal --- taken from CSC
  if( TGeoBBox* box = dynamic_cast<TGeoBBox*>( shape->GetShape()))     // Box         --- taken from ME0
  {

     LocalPoint pos = iData.localPosition();
     LocalVector dir = iData.localDirection();   
     LocalVector unit = dir.unit();
    
     Double_t localPosition[3]     = {  pos.x(),  pos.y(),  pos.z() };
     Double_t localDirectionIn[3]  = {  dir.x(),  dir.y(),  dir.z() };
     Double_t localDirectionOut[3] = { -dir.x(), -dir.y(), -dir.z() };
  
     // float distIn = trap->DistFromInside( localPosition, localDirectionIn );
     // float distOut = trap->DistFromInside( localPosition, localDirectionOut );
     float distIn = box->DistFromInside( localPosition, localDirectionIn );
     float distOut = box->DistFromInside( localPosition, localDirectionOut );
     LocalVector vIn = unit * distIn;
     LocalVector vOut = -unit * distOut;
     float localSegmentInnerPoint[3] = { static_cast<float>(localPosition[0] + vIn.x()),
					 static_cast<float>(localPosition[1] + vIn.y()),
					 static_cast<float>(localPosition[2] + vIn.z()) 
     };
      
     float localSegmentOuterPoint[3] = { static_cast<float>(localPosition[0] + vOut.x()),
					 static_cast<float>(localPosition[1] + vOut.y()),
					 static_cast<float>(localPosition[2] + vOut.z()) 
     };

     float globalSegmentInnerPoint[3];
     float globalSegmentOuterPoint[3];
     
     geom->localToGlobal( rawid, localSegmentInnerPoint,  globalSegmentInnerPoint, localSegmentOuterPoint,  globalSegmentOuterPoint );

     segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
			  globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );	

     std::cout<<"GEMSegment in DetId = "<<iData.gemDetId()<<" = "<<rawid<<" globalSegmentInnerPoint = ["
	      <<globalSegmentInnerPoint[0]<<","<<globalSegmentInnerPoint[1]<<","<<globalSegmentInnerPoint[2]<<"] "
	      <<"globalSegmentOuterPoint = ["<<globalSegmentOuterPoint[0]<<","<<globalSegmentOuterPoint[1]<<","<<globalSegmentOuterPoint[2]<<"]"<<std::endl;
  }
}

REGISTER_FWPROXYBUILDER( FWGEMSegmentProxyBuilder, GEMSegment, "GEM-segments", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


