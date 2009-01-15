// -*- C++ -*-
// $Id: FWHCalCaloTowerProxy3DBuilder.cc,v 1.1 2009/01/14 18:34:12 amraktad Exp $
//

#include "Fireworks/Calo/plugins/FWCaloTowerRPZProxyBuilderBase.h"

class FWHCalCaloTowerRPZProxyBuilder : public FWCaloTowerRPZProxyBuilderBase
{
public:
   FWHCalCaloTowerRPZProxyBuilder(): FWCaloTowerRPZProxyBuilderBase(false, "hcal3D") {}
   virtual ~FWHCalCaloTowerRPZProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHCalCaloTowerRPZProxyBuilder(const FWHCalCaloTowerRPZProxyBuilder&); // stop default

   const FWHCalCaloTowerRPZProxyBuilder& operator=(const FWHCalCaloTowerRPZProxyBuilder&); // stop default
};

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWHCalCaloTowerRPZProxyBuilder,CaloTowerCollection,"HCal");
