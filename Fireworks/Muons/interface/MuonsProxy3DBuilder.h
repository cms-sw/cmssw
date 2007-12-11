#ifndef Fireworks_Muons_MuonsProxy3DBuilder_h
#define Fireworks_Muons_MuonsProxy3DBuilder_h
// -*- C++ -*-
//  Description: muon model proxy
//  
//  Original Author: D.Kovalskyi
//
#include "Fireworks/Core/interface/FWDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

class MuonsProxy3DBuilder : public FWDataProxyBuilder
{
 public:
   MuonsProxy3DBuilder();
   virtual ~MuonsProxy3DBuilder();
   virtual void build(const fwlite::Event* iEvent, TEveElementList** oList);
 private:
   DetIdToMatrix detIdToMatrix_;
};

#endif
