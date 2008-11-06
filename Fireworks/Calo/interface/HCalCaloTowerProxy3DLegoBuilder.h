#ifndef Fireworks_Calo_HCalCaloTowerProxy3DLegoBuilder_h
#define Fireworks_Calo_HCalCaloTowerProxy3DLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     HCalCaloTowerProxy3DLegoBuilder
//
/**\class HCalCaloTowerProxy3DLegoBuilder HCalCaloTowerProxy3DLegoBuilder.h Fireworks/Calo/interface/HCalCaloTowerProxy3DLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: HCalCaloTowerProxy3DLegoBuilder.h,v 1.4 2008/07/07 00:32:36 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveHistProxyBuilder.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

// forward declarations
class TH2F;

class HCalCaloTowerProxy3DLegoBuilder : public FW3DLegoEveHistProxyBuilder
{

   public:
      HCalCaloTowerProxy3DLegoBuilder();
      virtual ~HCalCaloTowerProxy3DLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void applyChangesToAllModels();
      virtual void build(const FWEventItem* iItem,
			 TH2F** product);

      HCalCaloTowerProxy3DLegoBuilder(const HCalCaloTowerProxy3DLegoBuilder&); // stop default

      const HCalCaloTowerProxy3DLegoBuilder& operator=(const HCalCaloTowerProxy3DLegoBuilder&); // stop default

      // ---------- member data --------------------------------
      const CaloTowerCollection* m_towers;
      TH2F* m_hist;

};


#endif
