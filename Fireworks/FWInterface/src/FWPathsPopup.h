#ifndef Fireworks_FWInterface_FWPathsPopup_h
#define Fireworks_FWInterface_FWPathsPopup_h

#include "TGFrame.h"
#include <string>
#include <map>

namespace edm 
{
   class ScheduleInfo;
   class ModuleDescription;
   class Event;
   class EventSetup;
}

class FWFFLooper;

class TGLabel;
class TGTextEdit;
class TGTextButton;
class TGTextView;
class TGHtml;

class FWPathsPopup : public TGMainFrame
{
public:
   FWPathsPopup(FWFFLooper *);

   void postProcessEvent(edm::Event const&, edm::EventSetup const&);
   void postModule(edm::ModuleDescription const&);
   void scheduleReloadEvent();
   bool &hasChanges() { return m_hasChanges; };
   void setup(const edm::ScheduleInfo *info);
private:
   void makePathsTextView();
   void makePathsHtmlView();

   const edm::ScheduleInfo  *m_info;

#ifndef __CINT__
   FWFFLooper               *m_looper;
#endif
   bool                     m_hasChanges;


   TGLabel                  *m_moduleLabel;   
   TGLabel                  *m_moduleName;
   
   TGTextView               *m_modulePathsText;
   TGHtml                   *m_modulePathsHtml;

   TGTextEdit               *m_textEdit;
   TGTextButton             *m_apply;

   // Filled from ScheduleInfo
   std::vector<std::string> m_availableModuleLabels;
   std::vector<std::string> m_availablePaths;
   ClassDef(FWPathsPopup, 0);
};

#endif
