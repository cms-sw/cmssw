#ifndef Fireworks_Tracks_SiPixelProxy3DBuilder_h
#define Fireworks_Tracks_SiPixelProxy3DBuilder_h
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: SiPixelProxy3DBuilder.h,v 1.3 2008/07/20 18:28:02 dmytro Exp $
//

// system include files

class TEveElementList;
class FWEventItem;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Rtypes.h"
#include <vector>

class SiPixelProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      SiPixelProxy3DBuilder() {}
      virtual ~SiPixelProxy3DBuilder() {}
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      SiPixelProxy3DBuilder(const SiPixelProxy3DBuilder&); // stop default

      const SiPixelProxy3DBuilder& operator=(const SiPixelProxy3DBuilder&); // stop default
      
      void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
      void applyChangesToAllModels(TEveElement* iElements);
};

#endif
