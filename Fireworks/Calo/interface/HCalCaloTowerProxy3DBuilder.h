#ifndef Fireworks_Calo_HCalCaloTowerProxy3DBuilder_h
#define Fireworks_Calo_HCalCaloTowerProxy3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: HCalCaloTowerProxy3DBuilder.h,v 1.3 2008/07/01 19:16:43 chrjones Exp $
//

// system include files

class TEveElementList;
class FWEventItem;
class TEveCalo3D;

// user include files
#include "Fireworks/Calo/interface/ECalCaloTowerProxy3DBuilder.h"

class HCalCaloTowerProxy3DBuilder : public ECalCaloTowerProxy3DBuilder
{

   public:
      HCalCaloTowerProxy3DBuilder() { handleHcal(); }
      virtual ~HCalCaloTowerProxy3DBuilder() {}

      // ---------- const member functions ---------------------
      virtual std::string histName() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
   private:

      HCalCaloTowerProxy3DBuilder(const HCalCaloTowerProxy3DBuilder&); // stop default

      const HCalCaloTowerProxy3DBuilder& operator=(const HCalCaloTowerProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------
};

#endif
