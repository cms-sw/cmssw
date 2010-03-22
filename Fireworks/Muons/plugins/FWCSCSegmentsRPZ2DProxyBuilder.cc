// -*- C++ -*-
//
// Package:     Muons
// Class  :     CSCSegmentsProxyRhoPhiZ2DBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: FWCSCSegmentsRPZ2DProxyBuilder.cc,v 1.5 2010/02/26 10:28:40 eulisse Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

class FWCSCSegmentsRPZ2DProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<CSCSegment>
{

public:
   FWCSCSegmentsRPZ2DProxyBuilder();
   virtual ~FWCSCSegmentsRPZ2DProxyBuilder();
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCSCSegmentsRPZ2DProxyBuilder(const FWCSCSegmentsRPZ2DProxyBuilder&);
   const FWCSCSegmentsRPZ2DProxyBuilder& operator=(const FWCSCSegmentsRPZ2DProxyBuilder&);
  
  void buildRhoPhi(const CSCSegment& iData, 
                   unsigned int iIndex, TEveElement& oItemHolder) const;
  
  void buildRhoZ(const CSCSegment& iData,         
                 unsigned int iIndex, TEveElement& oItemHolder) const;
};

FWCSCSegmentsRPZ2DProxyBuilder::FWCSCSegmentsRPZ2DProxyBuilder()
{}

FWCSCSegmentsRPZ2DProxyBuilder::~FWCSCSegmentsRPZ2DProxyBuilder()
{}

void FWCSCSegmentsRPZ2DProxyBuilder::buildRhoPhi(const CSCSegment& iData,         
                                                 unsigned int iIndex, TEveElement& oItemHolder) const 
{}

void FWCSCSegmentsRPZ2DProxyBuilder::buildRhoZ(const CSCSegment& iData,           
                                               unsigned int iIndex, TEveElement& oItemHolder) const
{                                         
  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(iData.cscDetId().rawId());
  
  if (  ! matrix ) 
  {
    std::cout<<"ERROR: failed to get geometry of CSC chamber with detid: " 
             << iData.cscDetId().rawId() <<std::endl;
    return;
  }
  
  std::stringstream s;
  s << "chamber" << iIndex;
  
  TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
  segmentSet->SetLineWidth(3);
  segmentSet->SetMainColor(item()->defaultDisplayProperties().color());
  segmentSet->SetRnrSelf(item()->defaultDisplayProperties().isVisible());
  segmentSet->SetRnrChildren(item()->defaultDisplayProperties().isVisible());

  oItemHolder.AddElement(segmentSet);

  const double segmentLength = 15; // BAD! Hard-coded number
  
  double localSegmentInnerPoint[3];
  double localSegmentCenterPoint[3];
  double localSegmentOuterPoint[3];
  
  double globalSegmentInnerPoint[3];
  double globalSegmentCenterPoint[3];
  double globalSegmentOuterPoint[3];

  double localPositionX = iData.localPosition().x();
  double localPositionY = iData.localPosition().y();
  //double localPositionZ = iData.localPosition().z();

  double localDirectionX = iData.localDirection().x();
  double localDirectionY = iData.localDirection().y();
  double localDirectionZ = iData.localDirection().z();

  localSegmentOuterPoint[0] = localPositionX + segmentLength*localDirectionX;
  localSegmentOuterPoint[1] = localPositionY + segmentLength*localDirectionY;
  localSegmentOuterPoint[2] = segmentLength*localDirectionZ;

  localSegmentCenterPoint[0] = localPositionX;
  localSegmentCenterPoint[1] = localPositionY;
  localSegmentCenterPoint[2] = 0;

  localSegmentInnerPoint[0] = localPositionX - segmentLength*localDirectionX;
  localSegmentInnerPoint[1] = localPositionY - segmentLength*localDirectionY;
  localSegmentInnerPoint[2] = -segmentLength*localDirectionZ;

  matrix->LocalToMaster(localSegmentInnerPoint,  globalSegmentInnerPoint);
  matrix->LocalToMaster(localSegmentCenterPoint, globalSegmentCenterPoint);
  matrix->LocalToMaster(localSegmentOuterPoint,  globalSegmentOuterPoint);
  
  if ( globalSegmentInnerPoint[1]*globalSegmentOuterPoint[1] > 0 ) 
  {
    segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                        globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );       
  } 
  
  else 
  {  
    if ( fabs(globalSegmentInnerPoint[1]) > fabs(globalSegmentOuterPoint[1]) )
      segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                          globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2] );
    else
      segmentSet->AddLine(globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2],
                          globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
  }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWCSCSegmentsRPZ2DProxyBuilder, CSCSegment, "CSC-segments");
