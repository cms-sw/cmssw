// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWECalCaloTower3DProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 14:25:53 EST 2008
// $Id: FWECalCaloTower3DProxyBuilder.cc,v 1.1 2008/12/03 21:05:09 chrjones Exp $
//

// system include files
#include "DataFormats/CaloTowers/interface/CaloTower.h"

// user include files
#include "Fireworks/Calo/plugins/FWCaloTower3DProxyBuilderBase.h"


class FWECalCaloTower3DProxyBuilder : public FWCaloTower3DProxyBuilderBase {
   
public:
   FWECalCaloTower3DProxyBuilder() {}
   //virtual ~FWECalCaloTower3DProxyBuilder();
   
   // ---------- const member functions ---------------------
   virtual const std::string histName() const {
      return "ECal";
   }

   virtual double getEt(const CaloTower& iTower) const {
      return iTower.emEt();
   }

   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
private:
   FWECalCaloTower3DProxyBuilder(const FWECalCaloTower3DProxyBuilder&); // stop default
   
   const FWECalCaloTower3DProxyBuilder& operator=(const FWECalCaloTower3DProxyBuilder&); // stop default
   
   // ---------- member data --------------------------------
      
};

//
// static member functions
//
REGISTER_FW3DDATAPROXYBUILDER(FWECalCaloTower3DProxyBuilder,CaloTowerCollection,"ECal");
