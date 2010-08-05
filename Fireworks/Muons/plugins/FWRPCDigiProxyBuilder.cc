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
// $Id: FWRPCDigiProxyBuilder.cc,v 1.3 2010/07/28 09:47:19 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEvePointSet.h" // rm when done testing

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
    
    assert(parameters.size() == 3);

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

      TEvePointSet* testPointSet = new TEvePointSet();
      compound->AddElement(testPointSet);

      int strip = (*dit).strip();
      double centreOfStrip = (strip-0.5)*pitch + offset;
      
      double localPoint[3] = 
      {
        centreOfStrip, 0.0, 0.0
      };

      double globalPoint[3];

      matrix->LocalToMaster(localPoint, globalPoint);
    
      testPointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);

    }
  }
}

REGISTER_FWPROXYBUILDER(FWRPCDigiProxyBuilder, RPCDigiCollection, "RPCDigi", FWViewType::kISpyBit);

