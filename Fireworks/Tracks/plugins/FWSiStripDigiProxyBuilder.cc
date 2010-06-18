// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiStripDigiProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiStripDigiProxyBuilder.cc,v 1.9 2010/05/06 14:14:10 mccauley Exp $
//

#include "TEveCompound.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiStripDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWSiStripDigiProxyBuilder() {}
  virtual ~FWSiStripDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWSiStripDigiProxyBuilder(const FWSiStripDigiProxyBuilder&);    
  const FWSiStripDigiProxyBuilder& operator=(const FWSiStripDigiProxyBuilder&);
};

void FWSiStripDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const edm::DetSetVector<SiStripDigi>* digis = 0;

   iItem->get(digis);

   if ( ! digis )
   {
     fwLog(fwlog::kWarning)<<"ERROR: failed get SiStripDigis"<<std::endl;
     return;
   }
   
   for ( edm::DetSetVector<SiStripDigi>::const_iterator it = digis->begin(), end = digis->end();
         it != end; ++it)     
   { 
      TEveCompound* compound = new TEveCompound("si strip digi compound", "siStripDigis");
      compound->OpenCompound();
      product->AddElement(compound);
    
      edm::DetSet<SiStripDigi> ds = *it;
      const uint32_t& detID = ds.id;
      DetId detid(detID);
        
      for ( edm::DetSet<SiStripDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end();
            idigi != idigiEnd; ++idigi )        
      {
         TEvePointSet* pointSet = new TEvePointSet();
         pointSet->SetMarkerSize(2);
         pointSet->SetMarkerStyle(2);
         pointSet->SetMarkerColor(2);
         product->AddElement(pointSet);

         // For now, take the center of the strip as the local position 
         const DetIdToMatrix* detIdToGeo = iItem->getGeom();
         const TGeoHMatrix* matrix = detIdToGeo->getMatrix(detid);

         if ( ! matrix )
         {
           fwLog(fwlog::kWarning) 
             <<"ERROR: failed get geometry of SiStripDigi with detid: "
             << detid << std::endl;
           return;
         }
   
         double local[3] = 
           {
             0.0, 0.0, 0.0
           };
         
         double global[3] = 
           {
             0.0, 0.0, 0.0
           };
         
         matrix->LocalToMaster(local, global);
         pointSet->SetNextPoint(global[0], global[1], global[2]);

      } // end of iteration over digis  
   } // end of iteratin over the DetSetVector
}

REGISTER_FWPROXYBUILDER(FWSiStripDigiProxyBuilder,edm::DetSetVector<SiStripDigi>,"SiStripDigi", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
