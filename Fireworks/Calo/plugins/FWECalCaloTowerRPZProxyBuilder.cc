// -*- C++ -*-
// $Id: FWECalCaloTowerRPZProxyBuilder.cc,v 1.2 2009/01/14 18:34:12 amraktad Exp $
//

#include "Fireworks/Calo/plugins/FWCaloTowerRPZProxyBuilderBase.h"

class FWECalCaloTowerRPZProxyBuilder : public FWCaloTowerRPZProxyBuilderBase
{
public:
   FWECalCaloTowerRPZProxyBuilder(): FWCaloTowerRPZProxyBuilderBase(true, "ecal3D") {}
   virtual ~FWECalCaloTowerRPZProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWECalCaloTowerRPZProxyBuilder(const FWECalCaloTowerRPZProxyBuilder&); // stop default

   const FWECalCaloTowerRPZProxyBuilder& operator=(const FWECalCaloTowerRPZProxyBuilder&); // stop default
};

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWECalCaloTowerRPZProxyBuilder,CaloTowerCollection,"ECal");
