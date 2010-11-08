// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowNavigator_h
#define Fireworks_Core_CmsShowNavigator_h
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.h,v 1.48 2009/12/17 19:31:10 amraktad Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "Fireworks/Core/interface/FWFileEntry.h"
#include "TEventList.h"

// forward declarations
class TEventList;
class CSGAction;
class CmsShowMain;
class TFile;
class TGWindow;
class FWGUIEventFilter;

namespace edm {
   class EventID;
}

class CmsShowNavigator : public FWConfigurable
{
public:
   enum EFilterState { kOff, kOn, kWithdrawn };
   enum EFilterMode  { kOr = 1, kAnd = 2 };
   
private:
   typedef std::list<FWFileEntry*> FQBase_t;
   typedef FQBase_t::iterator      FQBase_i;


   struct FileQueue_t : public FQBase_t
   {
      struct iterator : public FQBase_i
      {
      private:
         bool m_isSet;

      public:
         iterator() : m_isSet(false) {}
         iterator(FQBase_i i) : FQBase_i(i), m_isSet(true) {}

         bool isSet() const { return m_isSet; }

         iterator& previous(FileQueue_t& cont)
         {
            // Go back one element, set to end() when falling off the end.
            if (*this == cont.begin())
               *this = cont.end();
            else
               FQBase_i::operator--();
            return *this;
         }
      };

      FileQueue_t() : FQBase_t() {}

      iterator begin() { return iterator(FQBase_t::begin()); }
      iterator end()   { return iterator(FQBase_t::end()); }
   };

   typedef FileQueue_t::iterator FileQueue_i;

public:
   CmsShowNavigator(const CmsShowMain &);
   virtual ~CmsShowNavigator();

   //configuration management interface
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);

   Int_t realEntry(Int_t rawEntry);
   bool  openFile(const std::string& fileName);
   bool  appendFile(const std::string& fileName, bool checkFileQueueSize, bool live);

   void nextEvent();
   void previousEvent();
   bool nextSelectedEvent();
   bool previousSelectedEvent();
   void firstEvent();
   void lastEvent();
   void goTo(FileQueue_i fi, int event);
   void goToRunEvent(Int_t,Int_t);

   void eventFilterEnableCallback(Bool_t);
   void filterEvents();
   void filterEventsAndReset();
  
   void setMaxNumberOfFilesToChain( unsigned int i ) {
      m_maxNumberOfFilesToChain = i;
   }
   bool isLastEvent();
   bool isFirstEvent();

   void showEventFilterGUI(const TGWindow* p);
   void applyFiltersFromGUI();
   void toggleFilterEnable();
   void withdrawFilter();
   void resumeFilter();
   
   const fwlite::Event* getCurrentEvent() const { return m_currentFile.isSet() ? (*m_currentFile)->event() : 0; }
   const char* filterStatusMessage();
   int  getNSelectedEvents();
   int  getNTotalEvents();
   bool canEditFiltersExternally();
   bool filesNeedUpdate() const { return m_filesNeedUpdate; }
   int  getFilterState() { return m_filterState; }
   
   void activateNewFileOnNextEvent() { m_newFileOnNextEvent = true; }
   void resetNewFileOnNextEvent()    { m_newFileOnNextEvent = false; }

   sigc::signal<void> newEvent_;
   sigc::signal<void, const TFile*> fileChanged_;
   sigc::signal<void> preFiltering_;
   sigc::signal<void> postFiltering_;
   sigc::signal<void, bool> editFiltersExternally_;
   sigc::signal<void, int> filterStateChanged_;
  

private:
   CmsShowNavigator(const CmsShowNavigator&);    // stop default
   const CmsShowNavigator& operator=(const CmsShowNavigator&);    // stop default

   void setCurrentFile(FileQueue_i);
   void updateFileFilters();
   void updateSelectorsInfo();

   void removeFilter(std::list<FWEventSelector*>::iterator);
   void addFilter(FWEventSelector*);
   void changeFilter(FWEventSelector*, bool filterNeedUpdate);

   void editFiltersExternally();
   void newFile(FileQueue_i);

   void setupMemoryInfo(int numEvents);
   void writeMemoryInfo();

   // ---------- member data --------------------------------
   
   std::list<FWEventSelector*>  m_selectors;
   FileQueue_t m_files;
   FileQueue_i m_currentFile;
   int m_currentEvent;

   EFilterState m_filterState;
   int          m_filterMode;
   bool         m_filesNeedUpdate;
   bool         m_newFileOnNextEvent;
   
   unsigned int m_maxNumberOfFilesToChain;
   // entry is an event index nubmer which runs from 0 to
   // #events or #selected_events depending on if we filter
   // events or not
   const CmsShowMain &m_main;
   FWGUIEventFilter*  m_guiFilter;

   // write per event memory usage
   int                  m_memoryInfoSamples;
   std::vector<Float_t> m_memoryResidentVec;
   std::vector<Float_t> m_memoryVirtualVec;
};

#endif
