#ifndef Fireworks_Calo_ECalCaloTowerProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_ECalCaloTowerProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ECalCaloTowerProxyRhoPhiZ2DBuilder
//
/**\class ECalCaloTowerProxyRhoPhiZ2DBuilder ECalCaloTowerProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: ECalCaloTowerProxyRhoPhiZ2DBuilder.h,v 1.7 2008/06/23 22:56:46 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/interface/BaseCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "TEveParamList.h"
#include <string>
// forward declarations

class TEveGeoShapeExtract;

class ECalCaloTowerProxyRhoPhiZ2DBuilder : public BaseCaloTowerProxyRhoPhiZ2DBuilder
{

   public:
      ECalCaloTowerProxyRhoPhiZ2DBuilder();
      virtual ~ECalCaloTowerProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
      static void buildCalo(const FWEventItem* iItem,
			       TEveElementList** product,
			       std::string name,
			       TEveCalo3D*& calo3d,
			       bool ecal);
      static std::vector<std::pair<double,double> > getThetaBins();
      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      ECalCaloTowerProxyRhoPhiZ2DBuilder(const ECalCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      const ECalCaloTowerProxyRhoPhiZ2DBuilder& operator=(const ECalCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
