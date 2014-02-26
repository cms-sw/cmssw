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
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <vector>

class FWDTSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<DTRecSegment4D>
{
public:
   FWDTSegmentProxyBuilder( void ) {}
   virtual ~FWDTSegmentProxyBuilder( void ) {}

  virtual bool haveSingleProduct() const override { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWDTSegmentProxyBuilder( const FWDTSegmentProxyBuilder& );
   const FWDTSegmentProxyBuilder& operator=( const FWDTSegmentProxyBuilder& );

   void buildViewType( const DTRecSegment4D& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type, const FWViewContext* ) override;
};

void
FWDTSegmentProxyBuilder::buildViewType( const DTRecSegment4D& iData,           
				unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type, const FWViewContext* )
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

      // In RhoZ view, draw segments at the middle of the chamber, otherwise they won't align with 1D rechits, 
      // for which only one coordinate is known.
      if (type == FWViewType::kRhoZ) { 
	localPosition[0]=0;
	localDirectionIn[0]=0;
	localDirectionOut[0]=0;
      }

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
      pointSet->SetMarkerSize(1.5);
      setupAddElement( pointSet, &oItemHolder );

      std::vector<DTRecHit1D> recHits;
      const DTChamberRecSegment2D* phiSeg = iData.phiSegment();      
      const DTSLRecSegment2D* zSeg = iData.zSegment();
      if (type == FWViewType::kRhoPhi && phiSeg) {
	recHits = phiSeg->specificRecHits();
      }
      if (type == FWViewType::kRhoZ && zSeg) {
	recHits = zSeg->specificRecHits();
      }

      for (std::vector<DTRecHit1D>::const_iterator rh=recHits.begin(); rh!=recHits.end(); ++rh){
	DTLayerId layerId = (*rh).wireId().layerId();
	LocalPoint hpos = (*rh).localPosition();
	float hitLocalPos[3]= {hpos.x(), hpos.y(), hpos.z()};
	if (layerId.superLayer()==2 && type == FWViewType::kRhoZ) {
	  // In RhoZ view, draw theta SL hits at the middle of the chamber, otherwise they won't align with 1D rechits, 
	  // for which only one coordinate is known.
	  hitLocalPos[1]=0;
	}
	float hitGlobalPoint[3];
	geom->localToGlobal(layerId, hitLocalPos, hitGlobalPoint);
	pointSet->SetNextPoint(hitGlobalPoint[0], hitGlobalPoint[1], hitGlobalPoint[2]);
      }
    }
  }
}

REGISTER_FWPROXYBUILDER( FWDTSegmentProxyBuilder, DTRecSegment4D, "DT-segments", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


