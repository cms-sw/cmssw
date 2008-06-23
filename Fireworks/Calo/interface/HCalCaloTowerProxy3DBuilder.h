#ifndef Fireworks_Calo_HCalCaloTowerProxy3DBuilder_h
#define Fireworks_Calo_HCalCaloTowerProxy3DBuilder_h
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: HCalCaloTowerProxy3DBuilder.h,v 1.1 2008/06/16 18:35:38 dmytro Exp $
//

// system include files

class TEveElementList;
class FWEventItem;
class TEveCalo3D;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class HCalCaloTowerProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      HCalCaloTowerProxy3DBuilder() { setHighPriority(true); }
      virtual ~HCalCaloTowerProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      HCalCaloTowerProxy3DBuilder(const HCalCaloTowerProxy3DBuilder&); // stop default

      const HCalCaloTowerProxy3DBuilder& operator=(const HCalCaloTowerProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------
};

#endif
