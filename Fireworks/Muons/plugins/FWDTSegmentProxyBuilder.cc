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
// $Id: FWDTSegmentProxyBuilder.cc,v 1.6 2010/05/12 10:35:27 mccauley Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Muons/interface/SegmentUtils.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

class FWDTSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<DTRecSegment4D>
{
public:
   FWDTSegmentProxyBuilder() {}
   virtual ~FWDTSegmentProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWDTSegmentProxyBuilder(const FWDTSegmentProxyBuilder&);
   const FWDTSegmentProxyBuilder& operator=(const FWDTSegmentProxyBuilder&);

   void build(const DTRecSegment4D& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void
FWDTSegmentProxyBuilder::build(const DTRecSegment4D& iData,           
                               unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
   const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(iData.chamberId().rawId());

   if (  ! matrix ) 
   {
      fwLog(fwlog::kError) << " failed to get geometry of DT chamber with detid: " 
                           << iData.chamberId().rawId() <<std::endl;
      return;
   }

   std::stringstream s;
   s << "chamber" << iIndex;

   TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
   segmentSet->SetLineWidth(3);
   setupAddElement(segmentSet, &oItemHolder);

   double localPosition[3] = 
     {
       iData.localPosition().x(), iData.localPosition().y(), iData.localPosition().z()
     };

   double localDirection[3] =
     {
       iData.localDirection().x(), iData.localDirection().y(), iData.localDirection().z()
     };

   double localSegmentInnerPoint[3];
   double localSegmentOuterPoint[3];

   fireworks::createSegment(MuonSubdetId::DT, false, 17.0, 
                            localPosition, localDirection, 
                            localSegmentInnerPoint, localSegmentOuterPoint);
                            
   double globalSegmentInnerPoint[3];
   double globalSegmentOuterPoint[3];

   matrix->LocalToMaster(localSegmentInnerPoint,  globalSegmentInnerPoint);
   matrix->LocalToMaster(localSegmentOuterPoint,  globalSegmentOuterPoint);

   segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                       globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2]);

}

REGISTER_FWPROXYBUILDER( FWDTSegmentProxyBuilder, DTRecSegment4D, "DT-segments", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);


