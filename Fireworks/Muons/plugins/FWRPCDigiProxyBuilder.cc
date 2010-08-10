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
// $Id: FWRPCDigiProxyBuilder.cc,v 1.4 2010/08/05 15:19:15 mccauley Exp $
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
    const RPCDetId& rpcDetId = (*dri).first;

    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(rpcDetId.rawId());
  
    if ( ! matrix ) 
    {
      fwLog(fwlog::kWarning)<<"Failed get geometry of RPC reference volume with detid: "
                            << rpcDetId.rawId() << std::endl;
      continue;
    }     
    
    std::vector<float> parameters = iItem->getGeom()->getParameters(rpcDetId.rawId());

    if ( parameters.empty() )
    {
      fwLog(fwlog::kWarning)<<"Parameters empty for RPC with detid: "
                            << rpcDetId.rawId() <<std::endl;
      continue;
    }
    
    assert(parameters.size() >= 3);

    float nStrips = parameters[0];
    float stripLength = parameters[1];
    float pitch = parameters[2];

    float offset = -0.5*nStrips*pitch;

    const RPCDigiCollection::Range& range = (*dri).second;

    for ( RPCDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )
    {
      TEveCompound* compound = new TEveCompound("rpc digi compound", "rpcDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet();
      stripDigiSet->SetLineWidth(3);
      compound->AddElement(stripDigiSet);

      int strip = (*dit).strip();
      double centreOfStrip = (strip-0.5)*pitch + offset;

      // Need to find actual intersection (for endcaps) but this should be 
      // good enough for debugging purposes 

      double localPointTop[3] =
      {
        centreOfStrip, stripLength*0.5, 0.0
      };

      double localPointBottom[3] = 
      {
        centreOfStrip, -stripLength*0.5, 0.0
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
                        FWViewType::kAll3DBits || FWViewType::kAllRPZBits);

