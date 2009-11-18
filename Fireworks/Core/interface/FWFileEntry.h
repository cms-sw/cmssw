// -*- C++ -*-
#ifndef Fireworks_Core_FWFileEntry_h
#define Fireworks_Core_FWFileEntryr_h
//
// Package:     Core
// Class  :     FWFileEntry
// $Id: FWFileEntry.h,v 1.2 2009/11/17 22:24:31 amraktad Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

#include "TEventList.h"
#include "TTree.h"

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
class TEventList;
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
      TEventList*        m_eventList;
      FWEventSelector*   m_selector;  // owned by navigator
      bool               m_needsUpdate;
      
      Filter(FWEventSelector* s) : m_eventList(0), m_selector(s), m_needsUpdate(true) {}
      ~Filter()
      {
         delete m_eventList;
      }
   };
   
   FWFileEntry(const std::string& name);
   virtual ~FWFileEntry();
   
   bool hasSelectedEvents() const {
      return m_eventTree && m_globalEventList->GetN()>0;
   }
   
   TFile*         file()  { return m_file; }
   fwlite::Event* event() { return m_event; }
   TTree*         tree()  { return m_eventTree; }
   TEventList*    globalSelection() { return m_globalEventList; }
   
   std::list<Filter*>& filters() { return m_filterEntries; }
   
   void openFile();
   void closeFile();

   bool isEventSelected(int event);

   bool hasSelectedEvents();

   bool hasActiveFilters();

   int  firstSelectedEvent();
   int  lastSelectedEvent();

   int  lastEvent() { return m_eventTree->GetEntries() -1; }

   int  nextSelectedEvent(int event);
   int  previousSelectedEvent(int event);

   void filtersNeedUpdate() { m_filtersNeedUpdate = true; }
   void updateFilters(FWEventItemsManager* eiMng, bool isOR);

private:
   FWFileEntry(const FWFileEntry&);    // stop default
   const FWFileEntry& operator=(const FWFileEntry&);    // stop default
   
   void runFilter(Filter* fe, FWEventItemsManager* eiMng);

   std::string            m_name;
   TFile*                 m_file;
   TTree*                 m_eventTree;
   fwlite::Event*         m_event;
   
   bool                   m_filtersNeedUpdate; // To be set in navigator::filterChanged/Added, newFile
   
   std::list<Filter*>     m_filterEntries;
   TEventList*            m_globalEventList;
};
#endif
