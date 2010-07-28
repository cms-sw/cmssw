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
// $Id: FWRPCDigiProxyBuilder.cc,v 1.1.2.7 2010/06/07 16:42:27 mccauley Exp $
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

  //std::cout<<"Got RPC digis"<<std::endl;

  for ( RPCDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {
    const RPCDetId& rpcDetId = (*dri).first;

    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(rpcDetId);
  
    if ( ! matrix ) 
    {
      std::cout << "ERROR: failed get geometry of RPC reference volume with detid: "
                << rpcDetId << std::endl;
      return;
    }     

    const RPCDigiCollection::Range& range = (*dri).second;

    /*
    std::cout<<"RPCDetId: "<< rpcDetId <<std::endl;
       
    int region = rpcDetId.region();
    int ring   = rpcDetId.ring();
    int station = rpcDetId.station();
    int sector = rpcDetId.station();
    int layer = rpcDetId.layer();
    int subsector = rpcDetId.subsector();
    int roll = rpcDetId.roll();
    */

    for ( RPCDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )
    {
      TEveCompound* compound = new TEveCompound("rpc digi compound", "rpcDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      /*
      int strip = (*dit).strip();
      int bx = (*dit).bx(); 

      std::cout<<"strip, bx: "<< strip <<" "<< bx <<std::endl;
      */
    }
  }
}

REGISTER_FWPROXYBUILDER(FWRPCDigiProxyBuilder, RPCDigiCollection, "RPCDigi", FWViewType::kISpyBit);

