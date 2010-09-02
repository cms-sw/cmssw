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
class TGHtml;
class TString;
class FWPSetTableManager;
class FWTableWidget;

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
   void makePathsView();

   const char* typeCodeToChar(char);
   void handlePSet(const edm::ParameterSet* ps, TString&);
   void handleEntry(const edm::Entry&, const std::string&, TString&);
   void handlePSetEntry(const edm::ParameterSetEntry&, const std::string&, TString&);
   void handleVPSetEntry(const edm::VParameterSetEntry&, const std::string&, TString&);

   const edm::ScheduleInfo  *m_info;

#ifndef __CINT__
   FWFFLooper               *m_looper;
#endif
   bool                     m_hasChanges;

   TGLabel                  *m_moduleLabel;   
   TGLabel                  *m_moduleName;
   
   TGHtml                   *m_modulePathsHtml;

   TGTextEdit               *m_textEdit;
   TGTextButton             *m_apply;
   FWPSetTableManager       *m_psTable;
   FWTableWidget            *m_tableWidget;

   // Filled from ScheduleInfo
   std::vector<std::string> m_availableModuleLabels;
   std::vector<std::string> m_availablePaths;
   ClassDef(FWPathsPopup, 0);
};

#endif
