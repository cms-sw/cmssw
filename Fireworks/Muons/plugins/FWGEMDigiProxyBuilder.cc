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
// $Id: FWRPCDigiProxyBuilder.cc,v 1.14 2010/09/07 15:46:48 yana Exp $
// $Id: FWGEMDigiProxyBuilder.cc,v 1.15 2013/10/10 22:06:00 YusangKim$
//

#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"

class FWGEMDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWGEMDigiProxyBuilder() {}
  virtual ~FWGEMDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  using FWProxyBuilderBase::build;
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWGEMDigiProxyBuilder(const FWGEMDigiProxyBuilder&);    
  const FWGEMDigiProxyBuilder& operator=(const FWGEMDigiProxyBuilder&);
};

void
FWGEMDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const GEMDigiCollection* digis = 0;
 
  iItem->get(digis);

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
    float topPitch = parameters[3];
    float bottomPitch = parameters[4];

    for( GEMDigiCollection::const_iterator dit = range.first;
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

REGISTER_FWPROXYBUILDER(FWGEMDigiProxyBuilder, GEMDigiCollection, "GEMDigi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);


class FWGEMPadDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWGEMPadDigiProxyBuilder() {}
  virtual ~FWGEMPadDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWGEMPadDigiProxyBuilder(const FWGEMPadDigiProxyBuilder&);    
  const FWGEMPadDigiProxyBuilder& operator=(const FWGEMPadDigiProxyBuilder&);
};

void
FWGEMPadDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const GEMPadDigiCollection* digis = 0;
 
  iItem->get(digis);

  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get GEMPadDigis"<<std::endl;
    return;
  }
  const FWGeometry *geom = iItem->getGeom();

  for ( GEMPadDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {
    unsigned int rawid = (*dri).first.rawId();
    const GEMPadDigiCollection::Range& range = (*dri).second;

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
    float nPads = parameters[5];
    float topPitch = parameters[3]*nStrips/nPads;
    float bottomPitch = parameters[4]*nStrips/nPads;

    for( GEMPadDigiCollection::const_iterator dit = range.first;
	 dit != range.second; ++dit )
    {
      TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet;
      stripDigiSet->SetLineWidth(3*nStrips/nPads);
      setupAddElement( stripDigiSet, product );

      int pad = (*dit).pad();
      float topOfStrip = (pad-0.5)*topPitch - 0.5*topPitch*nPads;
      float bottomOfStrip = (pad-0.5)*bottomPitch - 0.5*bottomPitch*nPads;

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

REGISTER_FWPROXYBUILDER(FWGEMPadDigiProxyBuilder, GEMPadDigiCollection, "GEMPadDigi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

