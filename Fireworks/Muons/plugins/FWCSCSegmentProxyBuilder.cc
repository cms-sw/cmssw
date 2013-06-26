// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWCSCSegmentProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCSCSegmentProxyBuilder.cc,v 1.19 2011/10/18 12:40:56 yana Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

class FWCSCSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<CSCSegment>
{
public:
  FWCSCSegmentProxyBuilder( void ) {}
  virtual ~FWCSCSegmentProxyBuilder( void ) {}
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  FWCSCSegmentProxyBuilder( const FWCSCSegmentProxyBuilder& );   
  const FWCSCSegmentProxyBuilder& operator=( const FWCSCSegmentProxyBuilder& );

  void build( const CSCSegment& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWCSCSegmentProxyBuilder::build( const CSCSegment& iData,           
				 unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  const FWGeometry *geom = item()->getGeom();
  unsigned int rawid = iData.cscDetId().rawId();
  
  if( ! geom->contains( rawid ))
  {
    fwLog(fwlog::kError) << "failed to get geometry of CSC chamber with rawid: " 
                         << rawid << std::endl;
    return;
  }
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet();
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 3 );
  setupAddElement( segmentSet, &oItemHolder );

  TEveGeoShape* shape = item()->getGeom()->getEveShape( rawid );
  if( TGeoTrap* trap = dynamic_cast<TGeoTrap*>( shape->GetShape())) // Trapezoidal
  {
     LocalPoint pos = iData.localPosition();
     LocalVector dir = iData.localDirection();   
     LocalVector unit = dir.unit();
    
     Double_t localPosition[3]     = {  pos.x(),  pos.y(),  pos.z() };
     Double_t localDirectionIn[3]  = {  dir.x(),  dir.y(),  dir.z() };
     Double_t localDirectionOut[3] = { -dir.x(), -dir.y(), -dir.z() };
  
     float distIn = trap->DistFromInside( localPosition, localDirectionIn );
     float distOut = trap->DistFromInside( localPosition, localDirectionOut );
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

REGISTER_FWPROXYBUILDER( FWCSCSegmentProxyBuilder, CSCSegment, "CSC-segments", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


