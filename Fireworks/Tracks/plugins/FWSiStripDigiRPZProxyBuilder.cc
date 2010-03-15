#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class FWSiStripDigiRPZProxyBuilder : public FWRPZDataProxyBuilder
{
public:
  FWSiStripDigiRPZProxyBuilder() {}
  virtual ~FWSiStripDigiRPZProxyBuilder() {}
  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList** product);
  FWSiStripDigiRPZProxyBuilder(const FWSiStripDigiRPZProxyBuilder&);    
  const FWSiStripDigiRPZProxyBuilder& operator=(const FWSiStripDigiRPZProxyBuilder&);
};

void FWSiStripDigiRPZProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
  TEveElementList* tList = *product;

  if( 0 == tList ) 
  {
    tList =  new TEveElementList(iItem->name().c_str(),"SiPixelDigi",true);
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
    gEve->AddElement(tList);
  } 
  
  else 
  {
    tList->DestroyElements();
  }

  const edm::DetSetVector<SiStripDigi>* digis = 0;
  iItem->get(digis);

  if( 0 == digis ) 
    return;

  for ( edm::DetSetVector<SiStripDigi>::const_iterator it = digis->begin(), end = digis->end();
        it != end; ++it)     
  {     
    TEveCompound* compound = new TEveCompound("si strip digi compound", "siStripDigis");
    compound->OpenCompound();
    tList->AddElement(compound);

    edm::DetSet<SiStripDigi> ds = *it;

    const uint32_t& detID = ds.id;
    DetId detid(detID);
      
    for ( edm::DetSet<SiStripDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end();
          idigi != idigiEnd; ++idigi )        
    {
      TEvePointSet* pointSet = new TEvePointSet();
      pointSet->SetMarkerSize(2);
      pointSet->SetMarkerStyle(2);
      pointSet->SetMarkerColor(46);
      tList->AddElement(pointSet);

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

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWSiStripDigiRPZProxyBuilder,edm::DetSetVector<SiStripDigi>,"SiStripDigi");
