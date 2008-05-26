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
// $Id: ECalCaloTowerProxyRhoPhiZ2DBuilder.h,v 1.3 2008/03/13 03:02:01 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/interface/BaseCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "TEveParamList.h"

// forward declarations

class TEveGeoShapeExtract;

class ECalCaloTowerProxyRhoPhiZ2DBuilder : public BaseCaloTowerProxyRhoPhiZ2DBuilder
{

   public:
      ECalCaloTowerProxyRhoPhiZ2DBuilder();
      virtual ~ECalCaloTowerProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      static TEveGeoShapeExtract* getRhoPhiElements(const char* name, 
						    const CaloTowerCollection* towers, 
						    Int_t color, 
						    bool hcal,
						    double eta_limit = 1.5,
						    double scale = 2 );
      static TEveGeoShapeExtract* getRhoZElements(  const char* name, 
						    const CaloTowerCollection* towers, 
						    Int_t color, 
						    bool hcal,
						    double scale = 2
						    );
      static TEveGeoShapeExtract* getRhoZElements(  const char* name, 
						    double size,
						    double r,
						    double theta,
						    double dTheta, 
						    Int_t color,
						    bool top);
      static std::vector<std::pair<double,double> > getThetaBins();
   
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
