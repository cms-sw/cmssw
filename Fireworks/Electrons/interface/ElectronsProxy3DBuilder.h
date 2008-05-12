#ifndef Fireworks_Electrons_ElectronsProxy3DBuilder_h
#define Fireworks_Electrons_ElectronsProxy3DBuilder_h
// -*- C++ -*-
//  Description: muon model proxy
//  
//  Original Author: D.Kovalskyi
//
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
class ElectronsProxy3DBuilder : public FWRPZDataProxyBuilder
{
 public:
   ElectronsProxy3DBuilder();
   virtual ~ElectronsProxy3DBuilder();

   static const reco::GsfElectronCollection *electrons;

 private:
   virtual void 	build (const FWEventItem* iItem, TEveElementList** product);
   ElectronsProxy3DBuilder (const ElectronsProxy3DBuilder&); // stop default
   const ElectronsProxy3DBuilder& operator=(const ElectronsProxy3DBuilder&); // stop default

};

#endif
