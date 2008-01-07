#ifndef Fireworks_Calo_ECalCaloTowerProxy3DLegoBuilder_h
#define Fireworks_Calo_ECalCaloTowerProxy3DLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ECalCaloTowerProxy3DLegoBuilder
// 
/**\class ECalCaloTowerProxy3DLegoBuilder ECalCaloTowerProxy3DLegoBuilder.h Fireworks/Calo/interface/ECalCaloTowerProxy3DLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations

class ECalCaloTowerProxy3DLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      ECalCaloTowerProxy3DLegoBuilder();
      virtual ~ECalCaloTowerProxy3DLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void build(const FWEventItem* iItem, 
			 TH2F** product);

      ECalCaloTowerProxy3DLegoBuilder(const ECalCaloTowerProxy3DLegoBuilder&); // stop default

      const ECalCaloTowerProxy3DLegoBuilder& operator=(const ECalCaloTowerProxy3DLegoBuilder&); // stop default

      // ---------- member data --------------------------------

};


#endif
