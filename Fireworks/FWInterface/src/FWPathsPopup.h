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
   class ParameterSet;
   class Entry;
   class ParameterSetEntry;
   class VParameterSetEntry;
}

class FWFFLooper;

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
   FWPathsPopup(FWFFLooper *);

   void postProcessEvent(edm::Event const&, edm::EventSetup const&);
   void postModule(edm::ModuleDescription const&);
   void scheduleReloadEvent();
   bool &hasChanges() { return m_hasChanges; };
   void setup(const edm::ScheduleInfo *info);
   void cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t iGlobalX, Int_t iGlobalY);
   void newIndexSelected(int);
   void updateFilterString(const char *str);

private:
   const edm::ScheduleInfo  *m_info;

#ifndef __CINT__
   FWFFLooper               *m_looper;
#endif
   bool                     m_hasChanges;

   TGLabel                  *m_moduleLabel;   
   TGLabel                  *m_moduleName;
   
   TGTextEdit               *m_textEdit;
   TGTextButton             *m_apply;
   FWPSetTableManager       *m_psTable;
   FWTableWidget            *m_tableWidget;
   TGTextEntry              *m_search;

   ClassDef(FWPathsPopup, 0);
};

#endif
