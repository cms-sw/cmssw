#ifndef Fireworks_FWInterface_FWPathsPopup_h
#define Fireworks_FWInterface_FWPathsPopup_h

#include "TGFrame.h"
#include <string>

namespace edm 
{
   class ScheduleInfo;
   class ModuleDescription;
   class Event;
   class EventSetup;
   class StreamContext;
   class ModuleCallingContext;
}

class FWFFLooper;
class FWGUIManager;

class TGLabel;
class TGTextEdit;
class TGTextButton;
class TString;
class FWPSetTableManager;
class FWTableWidget;
class TGTextEntry;

class FWPathsPopup : public TGMainFrame
{
public:
   FWPathsPopup(FWFFLooper *, FWGUIManager *);

   void postEvent(edm::Event const &event);
   void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
   void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
   void scheduleReloadEvent();
   bool &hasChanges() { return m_hasChanges; }
   void setup(const edm::ScheduleInfo *info);
   void applyEditor();
   void cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t iGlobalX, Int_t iGlobalY);
   void updateFilterString(const char *str);
   void windowIsClosing();

   Bool_t HandleKey(Event_t* event) override;

private:
   const edm::ScheduleInfo  *m_info;

#ifndef __CINT__
   FWFFLooper               *m_looper;
#endif
   bool                     m_hasChanges;

   TGLabel                  *m_moduleLabel;   
   TGLabel                   *m_moduleName;
   
   TGTextButton             *m_apply;
   FWPSetTableManager       *m_psTable;
   FWTableWidget            *m_tableWidget;
   TGTextEntry              *m_search;
   FWGUIManager             *m_guiManager;
   

   ClassDefOverride(FWPathsPopup, 0);
};

#endif
