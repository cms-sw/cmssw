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
// $Id: FWCSCSegmentProxyBuilder.cc,v 1.6 2010/04/22 12:54:48 mccauley Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Muons/interface/CSCUtils.h"

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

class FWCSCSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<CSCSegment>
{
public:
   FWCSCSegmentProxyBuilder() {}
   virtual ~FWCSCSegmentProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCSCSegmentProxyBuilder(const FWCSCSegmentProxyBuilder&);   
   const FWCSCSegmentProxyBuilder& operator=(const FWCSCSegmentProxyBuilder&);

  void build(const CSCSegment& iData, unsigned int iIndex, TEveElement& oItemHolder);
};

void
FWCSCSegmentProxyBuilder::build(const CSCSegment& iData,           
                                unsigned int iIndex, TEveElement& oItemHolder)
{
  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(iData.cscDetId().rawId());
  
  if ( ! matrix ) 
  {
    std::cout<<"ERROR: failed to get geometry of CSC chamber with detid: " 
             << iData.cscDetId().rawId() <<std::endl;
    return;
  }
 
  double length = 0.0;
  double thickness = 0.0;
 
  fireworks::fillCSCChamberParameters(iData.cscDetId().station(), 
                                      iData.cscDetId().ring(), 
                                      length, thickness);

  std::stringstream s;
  s << "chamber" << iIndex;
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
  segmentSet->SetLineWidth(3);
  setupAddElement(segmentSet, &oItemHolder);
  
  double localSegmentInnerPoint[3];
  double localSegmentOuterPoint[3];
  
  double globalSegmentInnerPoint[3];
  double globalSegmentOuterPoint[3];

  double localPositionX = iData.localPosition().x();
  double localPositionY = iData.localPosition().y();
  double localPositionZ = thickness*0.5;

  double localDirectionX = iData.localDirection().x();
  double localDirectionY = iData.localDirection().y();
  double localDirectionZ = iData.localDirection().z();

  localSegmentOuterPoint[0] = localPositionX + localDirectionX*(localPositionZ/localDirectionZ);
  localSegmentOuterPoint[1] = localPositionY + localDirectionY*(localPositionZ/localDirectionZ);
  localSegmentOuterPoint[2] = localPositionZ;

  length *= 0.5;

  if ( fabs(localSegmentOuterPoint[1]) > length )
    localSegmentOuterPoint[1] = length*(localSegmentOuterPoint[1]/fabs(localSegmentOuterPoint[1]));

  localSegmentInnerPoint[0] = localPositionX - localDirectionX*(localPositionZ/localDirectionZ);
  localSegmentInnerPoint[1] = localPositionY - localDirectionY*(localPositionZ/localDirectionZ);
  localSegmentInnerPoint[2] = -localPositionZ;

  if ( fabs(localSegmentInnerPoint[1]) > length )
    localSegmentInnerPoint[1] = length*(localSegmentInnerPoint[1]/fabs(localSegmentInnerPoint[1]));

  matrix->LocalToMaster(localSegmentInnerPoint,  globalSegmentInnerPoint);
  matrix->LocalToMaster(localSegmentOuterPoint,  globalSegmentOuterPoint);
  
  segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                      globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2]);
}

REGISTER_FWPROXYBUILDER( FWCSCSegmentProxyBuilder, CSCSegment, "CSC Segments", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit );


