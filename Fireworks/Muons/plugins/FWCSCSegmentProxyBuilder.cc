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
// $Id: FWCSCSegmentProxyBuilder.cc,v 1.13 2010/06/18 12:44:05 yana Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
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
  unsigned int rawid = iData.cscDetId().rawId();
  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( rawid );
  
  if( ! matrix ) 
  {
    fwLog(fwlog::kError) << "failed to get geometry of CSC chamber with rawid: " 
                         << rawid << std::endl;
    return;
  }
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet();
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 3 );
  setupAddElement( segmentSet, &oItemHolder );

  double length    = 0.0;
  double thickness = 0.0;

  TEveGeoShape* shape = item()->getGeom()->getShape( rawid );
  if( shape ) 
  {
    if( TGeoTrap* trap = dynamic_cast<TGeoTrap*>( shape->GetShape()))
    {
      length = trap->GetDz();
      thickness = trap->GetH1();

      LocalPoint pos = iData.localPosition();
      LocalVector dir = iData.localDirection();   
      LocalVector unit = dir.unit();
    
      double localPosition[3]     = {  pos.x(),  pos.y(),  pos.z() };
      double localDirectionIn[3]  = {  dir.x(),  dir.y(),  dir.z() };
      double localDirectionOut[3] = { -dir.x(), -dir.y(), -dir.z() };
  
      Double_t distIn = trap->DistFromInside( localPosition, localDirectionIn );
      Double_t distOut = trap->DistFromInside( localPosition, localDirectionOut );
      LocalVector vIn = unit * distIn;
      LocalVector vOut = -unit * distOut;
      double localSegmentInnerPoint[3] = { localPosition[0] + vIn.x(),
					   localPosition[1] + vIn.y(),
					   localPosition[2] + vIn.z() 
      };
      
      double localSegmentOuterPoint[3] = { localPosition[0] + vOut.x(),
					   localPosition[1] + vOut.y(),
					   localPosition[2] + vOut.z() 
      };

      double globalSegmentInnerPoint[3];
      double globalSegmentOuterPoint[3];

      matrix->LocalToMaster( localSegmentInnerPoint,  globalSegmentInnerPoint );
      matrix->LocalToMaster( localSegmentOuterPoint,  globalSegmentOuterPoint );

      segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
			   globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
    }
  }
}

REGISTER_FWPROXYBUILDER( FWCSCSegmentProxyBuilder, CSCSegment, "CSC-segments", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit );


