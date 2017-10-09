// -*- C++ -*-
#ifndef Fireworks_Core_FWFileEntry_h
#define Fireworks_Core_FWFileEntry_h
//
// Package:     Core
// Class  :     FWFileEntry
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

#include "TTree.h"

// user include files
// MT -- to get auxBranch
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWTEventList.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
class FWTEventList;
class CSGAction;
class CmsShowMain;
class TFile;
class TGWindow;
class FWEventItemsManager;

namespace edm {
   class EventID;
}

class FWFileEntry {
public:
   struct Filter
   {
      FWTEventList*      m_eventList;
      FWEventSelector*   m_selector;  // owned by navigator
      bool               m_needsUpdate;
      
      Filter(FWEventSelector* s) : m_eventList(0), m_selector(s), m_needsUpdate(true) {}
      ~Filter()
      {
         delete m_eventList;
      }

      bool hasSelectedEvents()
      {
         return m_eventList && m_eventList->GetN();
      }
   };
   
   FWFileEntry(const std::string& name, bool checkVersion);
   virtual ~FWFileEntry();
      
   TFile*         file()  { return m_file; }
   fwlite::Event* event() { return m_event; }
   TTree*         tree()  { return m_eventTree; }
   FWTEventList*    globalSelection() { return m_globalEventList; }
   
   std::list<Filter*>& filters() { return m_filterEntries; }
   
   void openFile(bool);
   void closeFile();

   bool isEventSelected(int event);

   bool hasSelectedEvents();

   bool hasActiveFilters();

   int  firstSelectedEvent();
   int  lastSelectedEvent();

   int  lastEvent() { return m_eventTree->GetEntries() -1; }

   int  nextSelectedEvent(int event);
   int  previousSelectedEvent(int event);

   void needUpdate() { m_needUpdate = true; }
   void updateFilters(const FWEventItemsManager* eiMng, bool isOR);

private:
   FWFileEntry(const FWFileEntry&);    // stop default
   const FWFileEntry& operator=(const FWFileEntry&);    // stop default
   
   void runFilter(Filter* fe, const FWEventItemsManager* eiMng);
   bool filterEventsWithCustomParser(Filter* filter);

   std::string            m_name;
   TFile*                 m_file;
   TTree*                 m_eventTree;
   fwlite::Event*         m_event;
   
   bool                   m_needUpdate; // To be set in navigator::filterChanged/Added, newFile
   
   std::list<Filter*>     m_filterEntries;
   FWTEventList*          m_globalEventList;
};
#endif
