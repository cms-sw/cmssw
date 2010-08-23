#ifndef Fireworks_FWInterface_FWPathsPopup_h
#define Fireworks_FWInterface_FWPathsPopup_h

#include "TGFrame.h"
#include <string>
#include <map>

namespace edm 
{
   class ScheduleInfo;
   class ModuleDescription;
}

class FWFFLooper;

class TGLabel;
class TGTextEdit;
class TGTextButton;

class FWPathsPopup : public TGMainFrame
{
public:
   FWPathsPopup(FWFFLooper *);

   void postModule(edm::ModuleDescription const&);
   void scheduleReloadEvent();
   bool &hasChanges() { return m_hasChanges; };
   void setup(const edm::ScheduleInfo *info);
private:
   const edm::ScheduleInfo  *m_info;
   TGLabel                  *m_moduleName;
   TGLabel                  *m_moduleLabel;
   TGTextEdit               *m_textEdit;
   TGTextButton             *m_apply;

#ifndef __CINT__
   FWFFLooper               *m_looper;
#endif
   bool                     m_hasChanges;
   ClassDef(FWPathsPopup, 0);
};

#endif
