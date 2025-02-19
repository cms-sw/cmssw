// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWDTSegmentProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWDTSegmentProxyBuilder.cc,v 1.14 2010/11/11 20:25:28 amraktad Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class FWDTSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<DTRecSegment4D>
{
public:
   FWDTSegmentProxyBuilder( void ) {}
   virtual ~FWDTSegmentProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWDTSegmentProxyBuilder( const FWDTSegmentProxyBuilder& );
   const FWDTSegmentProxyBuilder& operator=( const FWDTSegmentProxyBuilder& );

   void build( const DTRecSegment4D& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWDTSegmentProxyBuilder::build( const DTRecSegment4D& iData,           
				unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  unsigned int rawid = iData.chamberId().rawId();
  const FWGeometry *geom = item()->getGeom();

  if( ! geom->contains( rawid ))
  {
    fwLog( fwlog::kError ) << "failed to get geometry of DT chamber with detid: " 
			   << rawid << std::endl;
    return;
  }

  TEveStraightLineSet* segmentSet = new TEveStraightLineSet();
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 3 );
  setupAddElement( segmentSet, &oItemHolder );
   
  TEveGeoShape* shape = item()->getGeom()->getEveShape( rawid );
  if( shape ) 
  {
    if( TGeoBBox* box = dynamic_cast<TGeoBBox*>( shape->GetShape()))
    {
      LocalPoint pos = iData.localPosition();
      LocalVector dir = iData.localDirection();   
      LocalVector unit = dir.unit();
    
      double localPosition[3]     = {  pos.x(),  pos.y(),  pos.z() };
      double localDirectionIn[3]  = {  dir.x(),  dir.y(),  dir.z() };
      double localDirectionOut[3] = { -dir.x(), -dir.y(), -dir.z() };

      Double_t distIn = box->DistFromInside( localPosition, localDirectionIn );
      Double_t distOut = box->DistFromInside( localPosition, localDirectionOut );
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
    }
  }
}

REGISTER_FWPROXYBUILDER( FWDTSegmentProxyBuilder, DTRecSegment4D, "DT-segments", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


