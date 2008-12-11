#ifndef Fireworks_Tracks_SiPixelProxyPlain3DBuilder_h
#define Fireworks_Tracks_SiPixelProxyPlain3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: SiPixelProxyPlain3DBuilder.h,v 1.2 2008/11/06 22:05:30 amraktad Exp $
//

// system include files

class TEveElementList;
class FWEventItem;

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Rtypes.h"
#include <vector>

class SiPixelProxyPlain3DBuilder : public FW3DDataProxyBuilder
{
   public:
      SiPixelProxyPlain3DBuilder() {}
      virtual ~SiPixelProxyPlain3DBuilder() {}
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);
      SiPixelProxyPlain3DBuilder(const SiPixelProxyPlain3DBuilder&); // stop default
      const SiPixelProxyPlain3DBuilder& operator=(const SiPixelProxyPlain3DBuilder&); // stop default
      void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
      void applyChangesToAllModels(TEveElement* iElements);
};

#endif
