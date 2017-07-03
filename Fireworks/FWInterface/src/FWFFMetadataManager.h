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
   bool  hasModuleLabel(std::string& moduleLabel) override;

protected:
   bool doUpdate(FWJobMetadataUpdateRequest*) override;

private:
   const edm::Event* m_event;

};
#endif
