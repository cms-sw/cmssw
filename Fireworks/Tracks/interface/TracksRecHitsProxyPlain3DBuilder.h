#ifndef Fireworks_Muons_TracksRecHitsProxyPlain3DBuilder_h
#define Fireworks_Muons_TracksRecHitsProxyPlain3DBuilder_h
// $Id: TracksRecHitsProxyPlain3DBuilder.h,v 1.4 2008/11/06 22:05:23 amraktad Exp $

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"

class TracksRecHitsProxyPlain3DBuilder : public FW3DDataProxyBuilder
{

   public:
      TracksRecHitsProxyPlain3DBuilder() {}
      virtual ~TracksRecHitsProxyPlain3DBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);
      TracksRecHitsProxyPlain3DBuilder(const TracksRecHitsProxyPlain3DBuilder&); // stop default
      const TracksRecHitsProxyPlain3DBuilder& operator=(const TracksRecHitsProxyPlain3DBuilder&); // stop default

};

#endif
