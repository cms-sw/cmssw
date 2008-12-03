// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHCalCaloTower3DProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 14:25:53 EST 2008
// $Id$
//

// system include files
#include "DataFormats/CaloTowers/interface/CaloTower.h"

// user include files
#include "Fireworks/Calo/interface/FWCaloTower3DProxyBuilderBase.h"


class FWHCalCaloTower3DProxyBuilder : public FWCaloTower3DProxyBuilderBase {
   
public:
   FWHCalCaloTower3DProxyBuilder() {}
   //virtual ~FWHCalCaloTower3DProxyBuilder();
   
   // ---------- const member functions ---------------------
   virtual const std::string histName() const {
      return "HCal";
   }

   virtual double getEt(const CaloTower& iTower) const {
      return iTower.hadEt()+iTower.outerEt();
   }

   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
private:
   FWHCalCaloTower3DProxyBuilder(const FWHCalCaloTower3DProxyBuilder&); // stop default
   
   const FWHCalCaloTower3DProxyBuilder& operator=(const FWHCalCaloTower3DProxyBuilder&); // stop default
   
   // ---------- member data --------------------------------
      
};

//
// static member functions
//
REGISTER_FW3DDATAPROXYBUILDER(FWHCalCaloTower3DProxyBuilder,CaloTowerCollection,"HCal");
