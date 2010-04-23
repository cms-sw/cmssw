#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEvePointSet.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

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
  virtual void build(const FWEventItem* iItem, TEveElementList* product);
  FWSiStripDigiProxyBuilder(const FWSiStripDigiProxyBuilder&);    
  const FWSiStripDigiProxyBuilder& operator=(const FWSiStripDigiProxyBuilder&);
  void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
  void applyChangesToAllModels(TEveElement* iElements);
};

void FWSiStripDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product)
{
   const edm::DetSetVector<SiStripDigi>* digis = 0;
   iItem->get(digis);

   if( 0 == digis ) 
      return;

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
         double local[3]  = {0.0, 0.0, 0.0};
         double global[3] = {0.0, 0.0, 0.0};
         matrix->LocalToMaster(local, global);
         pointSet->SetNextPoint(global[0], global[1], global[2]);

      } // end of iteration over digis  
   } // end of iteratin over the DetSetVector
}

void
FWSiStripDigiProxyBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
FWSiStripDigiProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
{
  
}

REGISTER_FWPROXYBUILDER(FWSiStripDigiProxyBuilder,edm::DetSetVector<SiStripDigi>,"SiStripDigi", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
