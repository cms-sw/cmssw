#ifndef Fireworks_Tracks_SiStripProxy3DBuilder_h
#define Fireworks_Tracks_SiStripProxy3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: SiStripProxy3DBuilder.h,v 1.1 2008/08/20 23:59:52 dmytro Exp $
//

// system include files

class TEveElementList;
class FWEventItem;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Rtypes.h"
#include <vector>

class SiStripProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      SiStripProxy3DBuilder() {}
      virtual ~SiStripProxy3DBuilder() {}
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      SiStripProxy3DBuilder(const SiStripProxy3DBuilder&); // stop default

      const SiStripProxy3DBuilder& operator=(const SiStripProxy3DBuilder&); // stop default
      void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
      void applyChangesToAllModels(TEveElement* iElements);
};

#endif
