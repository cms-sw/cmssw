#ifndef Fireworks_Calo_HCalCaloTowerProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_HCalCaloTowerProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     HCalCaloTowerProxyRhoPhiZ2DBuilder
// 
/**\class HCalCaloTowerProxyRhoPhiZ2DBuilder HCalCaloTowerProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/HCalCaloTowerProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: HCalCaloTowerProxyRhoPhiZ2DBuilder.h,v 1.1 2008/02/03 02:57:10 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/interface/BaseCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

// forward declarations

class TEveGeoShapeExtract;

class HCalCaloTowerProxyRhoPhiZ2DBuilder : public BaseCaloTowerProxyRhoPhiZ2DBuilder
{

   public:
      HCalCaloTowerProxyRhoPhiZ2DBuilder();
      virtual ~HCalCaloTowerProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);
   
      virtual void buildRhoZ(const FWEventItem* iItem, 
                               TEveElementList** product);

      HCalCaloTowerProxyRhoPhiZ2DBuilder(const HCalCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      const HCalCaloTowerProxyRhoPhiZ2DBuilder& operator=(const HCalCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------

};


#endif
