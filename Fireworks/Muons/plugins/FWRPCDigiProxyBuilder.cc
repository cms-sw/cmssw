// -*- C++ -*-
//
// Package:     Muon
// Class  :     FWRPCDigiProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: mccauley
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRPCDigiProxyBuilder.cc,v 1.10 2010/08/17 15:21:42 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

class FWRPCDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWRPCDigiProxyBuilder() {}
  virtual ~FWRPCDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWRPCDigiProxyBuilder(const FWRPCDigiProxyBuilder&);    
  const FWRPCDigiProxyBuilder& operator=(const FWRPCDigiProxyBuilder&);
};

void
FWRPCDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const RPCDigiCollection* digis = 0;
 
  iItem->get(digis);

  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get RPCDigis"<<std::endl;
    return;
  }

  for ( RPCDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {
    unsigned int rawid = (*dri).first.rawId();
    float nStrips = 0.;
    float halfStripLength = 0.;
    float pitch = 0.;

    float offset = 0.;

    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(rawid);
    const float* parameters = iItem->getGeom()->getParameters(rawid);
 
    if( parameters == 0 )
    {
      fwLog(fwlog::kWarning)<<"Parameters empty for RPC with detid: "
			      << rawid <<std::endl;
    }    
    else
    {     
      nStrips = parameters[0];
      halfStripLength = parameters[1]*0.5;
      pitch = parameters[2];

      offset = -0.5*nStrips*pitch;
    }
    
    const RPCDigiCollection::Range& range = (*dri).second;

    for ( RPCDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )
    {
      TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet();
      stripDigiSet->SetLineWidth(3);
      setupAddElement(stripDigiSet, product);

      if ( ! matrix ) 
      {
	fwLog(fwlog::kWarning)<<"Failed get matrix of RPC with detid: "
			      << rawid << std::endl;
	continue;
      }     

      if ( parameters == 0 )
      {
        fwLog(fwlog::kWarning)<<"Parameters empty for RPC with detid: "
			      << rawid <<std::endl;
        continue;
      }
      
      int strip = (*dit).strip();
      double centreOfStrip = (strip-0.5)*pitch + offset;

      double localPointTop[3] =
      {
        centreOfStrip, halfStripLength, 0.0
      };

      double localPointBottom[3] = 
      {
        centreOfStrip, -halfStripLength, 0.0
      };

      double globalPointTop[3];
      double globalPointBottom[3];

      matrix->LocalToMaster(localPointTop, globalPointTop);
      matrix->LocalToMaster(localPointBottom, globalPointBottom);

      stripDigiSet->AddLine(globalPointTop[0], globalPointTop[1], globalPointTop[2],
                            globalPointBottom[0], globalPointBottom[1], globalPointBottom[2]);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWRPCDigiProxyBuilder, RPCDigiCollection, "RPCDigi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

