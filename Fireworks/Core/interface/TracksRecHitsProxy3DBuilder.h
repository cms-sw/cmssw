#ifndef Fireworks_Muons_TracksRecHitsProxy3DBuilder_h
#define Fireworks_Muons_TracksRecHitsProxy3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// Original name
// $Id: TracksRecHitsProxy3DBuilder.h,v 1.3 2008/08/20 23:52:36 dmytro Exp $
// New version
// $Id: TracksRecHitsProxy3DBuilder.h, v 1.0 2008 02/21 10:53:48 Tom Danielson
// system include files

class TEveElementList;
class FWEventItem;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

class TracksRecHitsProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      TracksRecHitsProxy3DBuilder() {}
      virtual ~TracksRecHitsProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      TracksRecHitsProxy3DBuilder(const TracksRecHitsProxy3DBuilder&); // stop default

      const TracksRecHitsProxy3DBuilder& operator=(const TracksRecHitsProxy3DBuilder&); // stop default

      void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
      void applyChangesToAllModels(TEveElement* iElements);
      // ---------- member data --------------------------------

};

#endif
