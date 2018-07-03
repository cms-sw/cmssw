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
  FWGEMSegmentProxyBuilder() {}
  ~FWGEMSegmentProxyBuilder() override {}
  
  bool haveSingleProduct() const override { return false; }

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWGEMSegmentProxyBuilder(const FWGEMSegmentProxyBuilder&) = delete;
  const FWGEMSegmentProxyBuilder& operator=(const FWGEMSegmentProxyBuilder&) = delete; 
 
  using FWSimpleProxyBuilderTemplate<GEMSegment>::buildViewType;
  void buildViewType(const GEMSegment& iData, 
                             unsigned int iIndex, 
                             TEveElement& oItemHolder, 
                             FWViewType::EType type, 
                             const FWViewContext*) override;
};


void
FWGEMSegmentProxyBuilder::buildViewType(const GEMSegment& iData,
					unsigned int iIndex, 
					TEveElement& oItemHolder, 
					FWViewType::EType type,
					const FWViewContext*)
{

  const FWGeometry *geom = item()->getGeom();
  unsigned int rawid = iData.gemDetId().rawId();
  
  if( !geom->contains( rawid )){
    fwLog(fwlog::kError) << "failed to get geometry of GEM chamber with rawid: " 
			 << rawid << std::endl;
    return;
  }
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet();
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 3 );
  segmentSet->SetMarkerColor(item()->defaultDisplayProperties().color());
  segmentSet->SetMarkerSize(0.5);
  setupAddElement( segmentSet, &oItemHolder );
  TEveGeoShape* shape = geom->getEveShape( rawid );
  if( shape )
    {
      if( TGeoTrap* box = dynamic_cast<TGeoTrap*>( shape->GetShape())) // Trapezoidal       
	{
	  shape->SetMainTransparency( 75 );
	  shape->SetMainColor( item()->defaultDisplayProperties().color());
	  segmentSet->AddElement( shape );
	  
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
	  if (type == FWViewType::kRhoPhi && std::abs(dir.x()) < 0.1){
	    segmentSet->AddMarker( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2] );
	  }
	  else 
	    segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
				 globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
	  
	}
    }
  
}

REGISTER_FWPROXYBUILDER( FWGEMSegmentProxyBuilder, GEMSegment, "GEM Segment", 
                         FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
