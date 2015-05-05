//#include "TEveStraightLineSet.h"
#include "TEveBoxSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"

class FWME0DigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWME0DigiProxyBuilder() {}
  virtual ~FWME0DigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWME0DigiProxyBuilder(const FWME0DigiProxyBuilder&);    
  const FWME0DigiProxyBuilder& operator=(const FWME0DigiProxyBuilder&);
};

void FWME0DigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const ME0DigiPreRecoCollection* digis = 0;
 
  iItem->get(digis);

  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get ME0Digis"<<std::endl;
    return;
  }
  const FWGeometry *geom = iItem->getGeom();

  for ( ME0DigiPreRecoCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {
    unsigned int rawid = (*dri).first.rawId();
    const ME0DigiPreRecoCollection::Range& range = (*dri).second;

    if( ! geom->contains( rawid ))
    {
      fwLog( fwlog::kWarning ) << "Failed to get geometry of ME0 roll with detid: "
                               << rawid << std::endl;
      
      TEveCompound* compound = createCompound();
      setupAddElement( compound, product );
      
      continue;
    }
    
    for( ME0DigiPreRecoCollection::const_iterator dit = range.first;
	 dit != range.second; ++dit )
    {
      TEveBoxSet* stripDigiSet = new TEveBoxSet;
      setupAddElement( stripDigiSet, product );
      
      float localPoint[3] =    {(*dit).x(), (*dit).y(), 0.0};
      float globalPoint[3];

      geom->localToGlobal( rawid, localPoint, globalPoint);

      stripDigiSet->AddBox(globalPoint[0], globalPoint[1], globalPoint[2],
                            (*dit).ex(), (*dit).ey(), 0.1);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWME0DigiProxyBuilder, ME0DigiPreRecoCollection, "ME0Digi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
