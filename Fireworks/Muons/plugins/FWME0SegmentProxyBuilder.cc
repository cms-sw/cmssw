#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TGeoArb8.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"

class FWME0SegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<ME0Segment>
{
public:
  FWME0SegmentProxyBuilder( void ) {}
  ~FWME0SegmentProxyBuilder( void ) override {}
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  FWME0SegmentProxyBuilder( const FWME0SegmentProxyBuilder& ) = delete;   
  const FWME0SegmentProxyBuilder& operator=( const FWME0SegmentProxyBuilder& ) = delete;

  void build( const ME0Segment& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWME0SegmentProxyBuilder::build( const ME0Segment& iData,           
				 unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  const FWGeometry *geom = item()->getGeom();
  unsigned int rawid = iData.me0DetId().rawId();
  
  if( ! geom->contains( rawid ))
  {
    fwLog(fwlog::kError) << "failed to get geometry of ME0 chamber with rawid: " 
                         << rawid << std::endl;
    return;
  }
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet();
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 5 );
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
      
      // Draw hits included in the segment
      TEvePointSet* pointSet = new TEvePointSet;
      // FIXME: This should be set elsewhere.
      pointSet->SetMarkerSize(0.5);
      pointSet->SetMarkerColor(1);
      setupAddElement( pointSet, &oItemHolder );
      auto recHits = iData.specificRecHits();
      for (auto rh = recHits.begin(); rh!= recHits.end(); rh++){
	auto me0id = rh->me0Id();
	LocalPoint hpos = rh->localPosition();
	float hitLocalPos[3]= {hpos.x(), hpos.y(), hpos.z()};
	float hitGlobalPoint[3];
	geom->localToGlobal(me0id, hitLocalPos, hitGlobalPoint);
	pointSet->SetNextPoint(hitGlobalPoint[0], hitGlobalPoint[1], hitGlobalPoint[2]);
      }
    }
  }
}

REGISTER_FWPROXYBUILDER( FWME0SegmentProxyBuilder, ME0Segment, "ME0-segments", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


