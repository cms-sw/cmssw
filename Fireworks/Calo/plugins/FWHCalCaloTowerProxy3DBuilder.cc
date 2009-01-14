// -*- C++ -*-
// $Id: FWHCalCaloTowerProxy3DBuilder.cc,v 1.1 2009/01/14 12:06:45 amraktad Exp $
//

#include "Fireworks/Calo/plugins/FWCaloTowerProxy3DBuilderBase.h"

class FWHCalCaloTowerProxy3DBuilder : public FWCaloTowerProxy3DBuilderBase
{
public:
   FWHCalCaloTowerProxy3DBuilder(): FWCaloTowerProxy3DBuilderBase(false, "hcal3D") {}
   virtual ~FWHCalCaloTowerProxy3DBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHCalCaloTowerProxy3DBuilder(const FWHCalCaloTowerProxy3DBuilder&); // stop default

   const FWHCalCaloTowerProxy3DBuilder& operator=(const FWHCalCaloTowerProxy3DBuilder&); // stop default
};

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWHCalCaloTowerProxy3DBuilder,CaloTowerCollection,"HCal");
