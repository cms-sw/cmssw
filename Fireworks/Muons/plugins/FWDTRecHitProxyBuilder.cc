// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWDTRecHitProxyBuilder
//
// $Id: FWDTRecHitProxyBuilder.cc,v 1.5 2010/04/20 20:49:43 amraktad Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

using namespace DTEnums;

class FWDTRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<DTRecHit1DPair>
{
public:
   FWDTRecHitProxyBuilder() {}
   virtual ~FWDTRecHitProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWDTRecHitProxyBuilder(const FWDTRecHitProxyBuilder&); 
   const FWDTRecHitProxyBuilder& operator=(const FWDTRecHitProxyBuilder&);

  void build(const DTRecHit1DPair& iData, unsigned int iIndex, TEveElement& oItemHolder);
};

void
FWDTRecHitProxyBuilder::build(const DTRecHit1DPair& iData,           
                              unsigned int iIndex, TEveElement& oItemHolder)
{
  DTChamberId chamberId(iData.geographicalId());

  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(chamberId);
    
   if( ! matrix ) 
   {
     std::cout <<"ERROR: failed get geometry of DT chamber with detid: " 
               << chamberId << std::endl;
     return;
   }

   std::stringstream s;
   s << "layer" << iIndex;

   TEveStraightLineSet* recHitSet = new TEveStraightLineSet(s.str().c_str());
   recHitSet->SetLineWidth(3);
   setupAddElement(recHitSet, &oItemHolder);

   double localCenterPoint[3] = 
     {
       0.0, 0.0, 0.0
     };
   
   double globalCenterPoint[3];
      
   double localPos[3] = 
     {
       iData.localPosition().x(), iData.localPosition().y(), iData.localPosition().z()
     };
   
   double globalPos[3];
	 
   const DTRecHit1D* lrechit = iData.componentRecHit(Left);
   const DTRecHit1D* rrechit = iData.componentRecHit(Right);

   double lLocalPos[3] = 
     {
       lrechit->localPosition().x(), lrechit->localPosition().y(), lrechit->localPosition().z()
     };
  
   double rLocalPos[3] = 
     {
       rrechit->localPosition().x(), rrechit->localPosition().y(), rrechit->localPosition().z()
     };

   double lGlobalPoint[3];
   double rGlobalPoint[3];

   matrix->LocalToMaster(lLocalPos, lGlobalPoint);
   matrix->LocalToMaster(rLocalPos, rGlobalPoint);
   matrix->LocalToMaster(localCenterPoint, globalCenterPoint);
   matrix->LocalToMaster(localPos, globalPos);
	 
   recHitSet->AddLine(lGlobalPoint[0], lGlobalPoint[1], lGlobalPoint[2], 
                      rGlobalPoint[0], rGlobalPoint[1], rGlobalPoint[2]);
   
   recHitSet->AddLine(globalCenterPoint[0], globalCenterPoint[1], globalCenterPoint[2], 
                      rGlobalPoint[0],      rGlobalPoint[1],      rGlobalPoint[2]);
   
   recHitSet->AddLine(lGlobalPoint[0],      lGlobalPoint[1],      lGlobalPoint[2], 
                      globalCenterPoint[0], globalCenterPoint[1], globalCenterPoint[2]);
   
   recHitSet->AddLine(globalPos[0],         globalPos[1],         globalPos[2], 
                      globalCenterPoint[0], globalCenterPoint[1], globalCenterPoint[2]);
}

REGISTER_FWPROXYBUILDER( FWDTRecHitProxyBuilder, DTRecHit1DPair, "DT RecHits", FWViewType::kISpyBit );


