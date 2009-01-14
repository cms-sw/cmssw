// -*- C++ -*-
// $Id: FWECalCaloTowerProxy3DBuilder.cc,v 1.1 2009/01/14 12:06:45 amraktad Exp $
//

#include "Fireworks/Calo/plugins/FWCaloTowerProxy3DBuilderBase.h"

class FWECalCaloTowerProxy3DBuilder : public FWCaloTowerProxy3DBuilderBase
{
public:
   FWECalCaloTowerProxy3DBuilder(): FWCaloTowerProxy3DBuilderBase(true, "ecal3D") {}
   virtual ~FWECalCaloTowerProxy3DBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWECalCaloTowerProxy3DBuilder(const FWECalCaloTowerProxy3DBuilder&); // stop default

   const FWECalCaloTowerProxy3DBuilder& operator=(const FWECalCaloTowerProxy3DBuilder&); // stop default
};

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWECalCaloTowerProxy3DBuilder,CaloTowerCollection,"ECal");
