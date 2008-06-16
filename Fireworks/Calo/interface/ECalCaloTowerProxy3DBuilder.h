#ifndef Fireworks_Calo_ECalCaloTowerProxy3DBuilder_h
#define Fireworks_Calo_ECalCaloTowerProxy3DBuilder_h
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: ECalCaloTowerProxy3DBuilder.h,v 1.2 2008/06/09 19:54:03 chrjones Exp $
//

// system include files

class TEveElementList;
class FWEventItem;
class TEveCalo3D;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class ECalCaloTowerProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      ECalCaloTowerProxy3DBuilder() {}
      virtual ~ECalCaloTowerProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      ECalCaloTowerProxy3DBuilder(const ECalCaloTowerProxy3DBuilder&); // stop default

      const ECalCaloTowerProxy3DBuilder& operator=(const ECalCaloTowerProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------
};

#endif
