#ifndef Fireworks_Muons_TracksProxy3DBuilder_h
#define Fireworks_Muons_TracksProxy3DBuilder_h
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: TracksProxy3DBuilder.h,v 1.1 2008/01/19 19:03:48 dmytro Exp $
//

// system include files

class TEveElementList;
class FWEventItem;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class TracksProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      TracksProxy3DBuilder() {}
      virtual ~TracksProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      TracksProxy3DBuilder(const TracksProxy3DBuilder&); // stop default

      const TracksProxy3DBuilder& operator=(const TracksProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------

};

#endif
