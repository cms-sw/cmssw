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
// $Id: FWDTSegmentProxyBuilder.cc,v 1.2 2010/04/16 16:40:13 yana Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class FWDTSegmentProxyBuilder : public FWSimpleProxyBuilderTemplate<DTRecSegment4D>
{
public:
   FWDTSegmentProxyBuilder() {}
   virtual ~FWDTSegmentProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWDTSegmentProxyBuilder(const FWDTSegmentProxyBuilder&);
   const FWDTSegmentProxyBuilder& operator=(const FWDTSegmentProxyBuilder&);

  void build(const DTRecSegment4D& iData, unsigned int iIndex, TEveElement& oItemHolder);
};

void
FWDTSegmentProxyBuilder::build(const DTRecSegment4D& iData,           
                               unsigned int iIndex, TEveElement& oItemHolder)
{
  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(iData.chamberId().rawId());

  if (  ! matrix ) 
  {
    std::cout<<"ERROR: failed to get geometry of DT chamber with detid: " 
             << iData.chamberId().rawId() <<std::endl;
    return;
  }

  std::stringstream s;
  s << "chamber" << iIndex;

  TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
  segmentSet->SetLineWidth(3);
  setupAddElement(segmentSet, &oItemHolder);

  const double halfThickness = 17.0; 
  // Bad! Actually the DTs are of either halfThickness 18.1 or 16.35
  // This should be fetched from the geometry.

  double localSegmentInnerPoint[3];
  double localSegmentOuterPoint[3];
  
  double globalSegmentInnerPoint[3];
  double globalSegmentOuterPoint[3];

  double localPositionX = iData.localPosition().x();
  double localPositionY = iData.localPosition().y();
  double localPositionZ = iData.localPosition().z();

  double localDirectionX = iData.localDirection().x();
  double localDirectionY = iData.localDirection().y();
  double localDirectionZ = iData.localDirection().z();

  double localDirMag = sqrt(localDirectionX*localDirectionX + 
                            localDirectionY*localDirectionY +
                            localDirectionZ*localDirectionZ);
  double localDirTheta = iData.localDirection().theta();

  localSegmentInnerPoint[0] = localPositionX + (localDirectionX/localDirMag)*(halfThickness/cos(localDirTheta));
  localSegmentInnerPoint[1] = localPositionY + (localDirectionY/localDirMag)*(halfThickness/cos(localDirTheta));
  localSegmentInnerPoint[2] = localPositionZ + (localDirectionZ/localDirMag)*(halfThickness/cos(localDirTheta));

  localSegmentOuterPoint[0] = localPositionX - (localDirectionX/localDirMag)*(halfThickness/cos(localDirTheta));
  localSegmentOuterPoint[1] = localPositionY - (localDirectionY/localDirMag)*(halfThickness/cos(localDirTheta));
  localSegmentOuterPoint[2] = localPositionZ - (localDirectionZ/localDirMag)*(halfThickness/cos(localDirTheta));

  matrix->LocalToMaster(localSegmentInnerPoint,  globalSegmentInnerPoint);
  matrix->LocalToMaster(localSegmentOuterPoint,  globalSegmentOuterPoint);

  segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                      globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2]);

}

REGISTER_FWPROXYBUILDER( FWDTSegmentProxyBuilder, DTRecSegment4D, "DT Segments", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);


