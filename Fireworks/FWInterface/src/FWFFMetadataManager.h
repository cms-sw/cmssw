#ifndef Fireworks_FWInterface_FWFFMetadataManager_h
#define Fireworks_FWInterface_FWFFMetadataManager_h

#include "Fireworks/Core/interface/FWJobMetadataManager.h"

class FWJobMetadataUpdateRequest;

class FWFFMetadataManager : public FWJobMetadataManager
{
public:
   // FIXME: does nothing for the time being!
protected:
   virtual bool doUpdate(FWJobMetadataUpdateRequest*);
};
#endif
