#ifndef Fireworks_Core_FWLiteJobMetadataManager
#define Fireworks_Core_FWLiteJobMetadataManager

#include "Fireworks/Core/interface/FWJobMetadataManager.h"

namespace fwlite
{
   class Event;
}

class FWJobMetadataUpdateRequest;

class FWLiteJobMetadataManager : public FWJobMetadataManager
{
public:
   FWLiteJobMetadataManager(void);
   virtual bool doUpdate(FWJobMetadataUpdateRequest *request);

   virtual bool  hasModuleLabel(std::string& moduleLabel);

private:
   const fwlite::Event *m_event;
};

#endif
