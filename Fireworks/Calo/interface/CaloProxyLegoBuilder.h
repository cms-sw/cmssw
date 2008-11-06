#ifndef Fireworks_Calo_CaloProxyLegoBuilder_h
#define Fireworks_Calo_CaloProxyLegoBuilder_h
// -*- C++ -*-
//  Description: muon model proxy
//
//  Original Author: D.Kovalskyi
//
#include "Fireworks/Core/interface/FWDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

class CaloProxyLegoBuilder : public FWDataProxyBuilder
{
 public:
   CaloProxyLegoBuilder();
   virtual ~CaloProxyLegoBuilder();
   virtual void build(const fwlite::Event* iEvent, TObject** product);
 private:
   double deltaR( double, double, double, double );
   unsigned int m_legoRebinFactor;
};

#endif
