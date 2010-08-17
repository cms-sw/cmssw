#ifndef Fireworks_FWInterface_FWPathsPopup_h
#define Fireworks_FWInterface_FWPathsPopup_h
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "TGFrame.h"

namespace edm 
{
   class ScheduleInfo;
}

class TGLabel;

class FWPathsPopup : public TGMainFrame
{
public:
   FWPathsPopup(const edm::ScheduleInfo *info);
   
   void postModule(edm::ModuleDescription const&);
private:
   const edm::ScheduleInfo  *m_info;
   TGLabel                  *m_moduleName;
   TGLabel                  *m_moduleLabel;
};

#endif
