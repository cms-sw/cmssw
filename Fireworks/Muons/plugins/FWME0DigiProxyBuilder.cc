#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
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
  const ME0DigiCollection* digis = 0;
 
  iItem->get(digis);

  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get ME0Digis"<<std::endl;
    return;
  }
  const FWGeometry *geom = iItem->getGeom();

  for ( ME0DigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {
    unsigned int rawid = (*dri).first.rawId();
    const ME0DigiCollection::Range& range = (*dri).second;

    if( ! geom->contains( rawid ))
    {
      fwLog( fwlog::kWarning ) << "Failed to get geometry of ME0 roll with detid: "
                               << rawid << std::endl;
      
      TEveCompound* compound = createCompound();
      setupAddElement( compound, product );
      
      continue;
    }

    const float* parameters = geom->getParameters( rawid );
    float nStrips = parameters[0];
    float halfStripLength = parameters[1]*0.5;
    float topPitch = parameters[3];
    float bottomPitch = parameters[4];

    for( ME0DigiCollection::const_iterator dit = range.first;
	 dit != range.second; ++dit )
    {
      TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet;
      stripDigiSet->SetLineWidth(3);
      setupAddElement( stripDigiSet, product );

      int strip = (*dit).strip();
      float topOfStrip = (strip-0.5)*topPitch - 0.5*nStrips*topPitch;
      float bottomOfStrip = (strip-0.5)*bottomPitch - 0.5*nStrips*bottomPitch;

      float localPointTop[3] =
      {
        topOfStrip, halfStripLength, 0.0
      };

      float localPointBottom[3] = 
      {
        bottomOfStrip, -halfStripLength, 0.0
      };

      float globalPointTop[3];
      float globalPointBottom[3];

      geom->localToGlobal( rawid, localPointTop, globalPointTop, localPointBottom, globalPointBottom );

      stripDigiSet->AddLine(globalPointTop[0], globalPointTop[1], globalPointTop[2],
                            globalPointBottom[0], globalPointBottom[1], globalPointBottom[2]);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWME0DigiProxyBuilder, ME0DigiCollection, "ME0Digi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
