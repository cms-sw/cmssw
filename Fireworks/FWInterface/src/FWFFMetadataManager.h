#ifndef Fireworks_FWInterface_FWFFMetadataManager_h
#define Fireworks_FWInterface_FWFFMetadataManager_h

#include "Fireworks/Core/interface/FWJobMetadataManager.h"

namespace edm {
class Event;
}
class FWJobMetadataUpdateRequest;

class FWFFMetadataManager : public FWJobMetadataManager
{
public:
   FWFFMetadataManager();
   virtual bool  hasModuleLabel(std::string& moduleLabel);

protected:
   virtual bool doUpdate(FWJobMetadataUpdateRequest*);

private:
   const edm::Event* m_event;

};
#endif
