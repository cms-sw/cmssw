#ifndef Fireworks_Calo_ECalCaloTowerProxy3DBuilder_h
#define Fireworks_Calo_ECalCaloTowerProxy3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: ECalCaloTowerProxy3DBuilder.h,v 1.4 2008/07/09 20:04:29 chrjones Exp $
//

// system include files
#include "Rtypes.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

class TEveElementList;
class FWEventItem;
class TEveCalo3D;
class TH2F;
class TEveCaloDataHist;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class ECalCaloTowerProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      ECalCaloTowerProxy3DBuilder():m_hist(0), m_sliceIndex(-1), m_handleEcal(true), m_towers(0) { setHighPriority( true ); }
      virtual ~ECalCaloTowerProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();
   protected:
      void handleHcal() {
         m_handleEcal=false;
      }

      virtual std::string histName() const;

      void itemBeingDestroyedImp(const FWEventItem*);

   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      virtual void modelChanges(const FWModelIds& iIds,
                                TEveElement* iElements );
      virtual void applyChangesToAllModels(TEveElement* iElements);
   ECalCaloTowerProxy3DBuilder(const ECalCaloTowerProxy3DBuilder&); // stop default

      const ECalCaloTowerProxy3DBuilder& operator=(const ECalCaloTowerProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------
      TH2F* m_hist;
      static TEveCaloDataHist* m_data;
      Int_t m_sliceIndex;
      bool m_handleEcal;
     const CaloTowerCollection* m_towers;
};

#endif
