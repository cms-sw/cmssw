#ifndef Fireworks_Core_ContextFF_h
#define Fireworks_Core_ContextFF_h
#include "Fireworks/Core/interface/Context.h"

namespace edm {
class EventSetup;
}

namespace fireworks {
class ContextFF : public Context {
public:
 ContextFF(FWModelChangeManager* iCM,
           FWSelectionManager*   iSM,
           FWEventItemsManager*  iEM,
           FWColorManager*       iColorM,
           FWJobMetadataManager* iJMDM);

   void setEventSetup(const edm::EventSetup* x) { m_setup = x;}
   const edm::EventSetup* getEventSetup() const { return m_setup; }

private:
   const edm::EventSetup* m_setup;
   
};
}
#endif
