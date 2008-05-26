#ifndef Fireworks_Muons_MuonsProxy3DBuilder_h
#define Fireworks_Muons_MuonsProxy3DBuilder_h
// -*- C++ -*-
//  Description: muon model proxy
//  
//  Original Author: D.Kovalskyi
//
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

namespace reco
{
   class Muon;
}

class TEveTrack;
class TEveElementList;

class MuonsProxy3DBuilder : public FWRPZDataProxyBuilder
{
 public:
   MuonsProxy3DBuilder();
   virtual ~MuonsProxy3DBuilder();
	
 private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);
   
   MuonsProxy3DBuilder(const MuonsProxy3DBuilder&); // stop default
   
   const MuonsProxy3DBuilder& operator=(const MuonsProxy3DBuilder&); // stop default

};

#endif
