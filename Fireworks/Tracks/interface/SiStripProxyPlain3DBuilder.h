#ifndef Fireworks_Tracks_SiStripProxyPlain3DBuilder_h
#define Fireworks_Tracks_SiStripProxyPlain3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: SiStripProxyPlain3DBuilder.h,v 1.2 2008/11/06 22:05:30 amraktad Exp $
//

// system include files

class TEveElementList;
class FWEventItem;

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Rtypes.h"
#include <vector>

class SiStripProxyPlain3DBuilder : public FW3DDataProxyBuilder
{
   public:
      SiStripProxyPlain3DBuilder() {}
      virtual ~SiStripProxyPlain3DBuilder() {}
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);
      SiStripProxyPlain3DBuilder(const SiStripProxyPlain3DBuilder&); // stop default
      const SiStripProxyPlain3DBuilder& operator=(const SiStripProxyPlain3DBuilder&); // stop default
      void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
      void applyChangesToAllModels(TEveElement* iElements);
};

#endif
