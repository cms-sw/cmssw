// -*- C++ -*-
//
// Package:     Muon
// Class  :     FWGEMDigiProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: mccauley
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWGEMDigiProxyBuilder.cc,v 1.13 2010/09/06 15:49:55 yana Exp $
//

#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "TEveManager.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class FWGEMDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWGEMDigiProxyBuilder() {}
  virtual ~FWGEMDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWGEMDigiProxyBuilder(const FWGEMDigiProxyBuilder&);    
  const FWGEMDigiProxyBuilder& operator=(const FWGEMDigiProxyBuilder&);
};

void
FWGEMDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{

  const GEMDigiCollection* digis = 0;
 
  iItem->get(digis);

  printf("GEMDigiCollection size (int)%d \n", (int)iItem->size());
  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get GEMDigis"<<std::endl;
    return;
  }
  const FWGeometry *geom = iItem->getGeom();

  for ( GEMDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {
    unsigned int rawid = (*dri).first.rawId();
    printf("GEMDigi [%s] ==> rawId = %d \n", item()->name().c_str(), rawid );
    const GEMDigiCollection::Range& range = (*dri).second;

    if( ! geom->contains( rawid ))
    {
      fwLog( fwlog::kWarning ) << "Failed to get geometry of GEM roll with detid: "
			       << rawid << std::endl;
      
      TEveCompound* compound = createCompound();
      setupAddElement( compound, product );
      
      continue;
    }
    
    const float* parameters = geom->getParameters( rawid );
    float nStrips = parameters[0];
    float halfStripLength = parameters[1]*0.5;
    float pitch = parameters[2];
    float offset = -0.5*nStrips*pitch;
    
    for( GEMDigiCollection::const_iterator dit = range.first;
	 dit != range.second; ++dit )
    {
      TEvePointSet* ps = new TEvePointSet("tmp", 1);
      ps->SetMarkerStyle(2);
      ps->SetMarkerSize(2);
      TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet;
      stripDigiSet->SetLineWidth(3);



      int strip = (*dit).strip();
      float centreOfStrip = (strip-0.5)*pitch + offset;

      float localPointTop[3] =
      {
        centreOfStrip, halfStripLength, 0.0
      };

      float localPointBottom[3] = 
      {
        centreOfStrip, -halfStripLength, 0.0
      };

      float globalPointTop[3];
      float globalPointBottom[3];

      geom->localToGlobal( rawid, localPointTop, globalPointTop, localPointBottom, globalPointBottom );

      printf("Strip = %d global pnts...(%f, %f, %f) (%f, %f, %f)\n", strip,globalPointTop[0], globalPointTop[1], globalPointTop[2],
                            globalPointBottom[0], globalPointBottom[1], globalPointBottom[2]);

      stripDigiSet->AddLine(globalPointTop[0], globalPointTop[1], globalPointTop[2],
                            globalPointBottom[0], globalPointBottom[1], globalPointBottom[2]);

      // debug == draw marker at the first point in case length is zero
      ps->SetNextPoint(globalPointTop[0], globalPointTop[1], globalPointTop[2]);
      ps->SetNextPoint(globalPointBottom[0], globalPointBottom[1], globalPointBottom[2]);

      setupAddElement( stripDigiSet, product );
      setupAddElement( ps, product );
    }
  }
}

REGISTER_FWPROXYBUILDER(FWGEMDigiProxyBuilder, GEMDigiCollection, "GEMDigi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

