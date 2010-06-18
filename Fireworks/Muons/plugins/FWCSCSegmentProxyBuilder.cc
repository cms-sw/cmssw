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
// $Id: FWCSCSegmentProxyBuilder.cc,v 1.12 2010/05/21 13:45:46 mccauley Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Muons/interface/CSCUtils.h"
#include "Fireworks/Muons/interface/SegmentUtils.h"

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

class FWCSCSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<CSCSegment>
{
public:
   FWCSCSegmentProxyBuilder() {}
   virtual ~FWCSCSegmentProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCSCSegmentProxyBuilder(const FWCSCSegmentProxyBuilder&);   
   const FWCSCSegmentProxyBuilder& operator=(const FWCSCSegmentProxyBuilder&);

  void build(const CSCSegment& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void
FWCSCSegmentProxyBuilder::build(const CSCSegment& iData,           
                                unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(iData.cscDetId().rawId());
  
  if ( ! matrix ) 
  {
    fwLog(fwlog::kError) << " failed to get geometry of CSC chamber with detid: " 
                         << iData.cscDetId().rawId() <<std::endl;
    return;
  }
  
  double length    = 0.0;
  double thickness = 0.0;
 
  fireworks::fillCSCChamberParameters(iData.cscDetId().station(), 
                                      iData.cscDetId().ring(), 
                                      length, thickness);

  std::stringstream s;
  s << "chamber" << iIndex;
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
  segmentSet->SetLineWidth(3);
  setupAddElement(segmentSet, &oItemHolder);
  
  double localPosition[3] = 
  {
    iData.localPosition().x(), iData.localPosition().y(), thickness*0.5
  };
  
  length *= 0.5;

  // If there is a bad reconstruction of the CSC segment one can have positions that
  // lie outside the chamber. In this case just drawn the position but do not
  // attempt to draw a line.

  // Q: can we determine this from the shape? A: It would make sense, but doesn't work when 
  // trying to determine from TGeoShape

  if ( fabs(localPosition[1]) > length )
  {
    fwLog(fwlog::kWarning) <<" position of CSC segment "<< iIndex <<" lies outside the chamber; station: "
                           << iData.cscDetId().station() <<"  ring: "<< iData.cscDetId().ring()
                           << std::endl;  
    return;
  }

  double localDirection[3] =
  {
    iData.localDirection().x(), iData.localDirection().y(), iData.localDirection().z()
  };
  
  double localSegmentInnerPoint[3];
  double localSegmentOuterPoint[3];
 
  fireworks::createSegment(MuonSubdetId::CSC, false, 
                           length, 0.0,
                           localPosition, localDirection,
                           localSegmentInnerPoint, localSegmentOuterPoint);

  double globalSegmentInnerPoint[3];
  double globalSegmentOuterPoint[3];

  matrix->LocalToMaster(localSegmentInnerPoint,  globalSegmentInnerPoint);
  matrix->LocalToMaster(localSegmentOuterPoint,  globalSegmentOuterPoint);
  
  segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                      globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2]);
}

REGISTER_FWPROXYBUILDER( FWCSCSegmentProxyBuilder, CSCSegment, "CSC-segments", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit );


