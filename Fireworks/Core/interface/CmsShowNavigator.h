// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowNavigator_h
#define Fireworks_Core_CmsShowNavigator_h
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.h,v 1.56 2012/07/06 23:33:29 amraktad Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

// user include files
#include "Fireworks/Core/interface/FWNavigatorBase.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "Fireworks/Core/interface/FWFileEntry.h"

#include "DataFormats/FWLite/interface/Event.h"

#include "TEventList.h"

// forward declarations
class TEventList;
class CSGAction;
class CmsShowMain;
class TFile;
class TGWindow;
class FWGUIEventFilter;

namespace edm {
   class EventBase;
   class EventID;
}

class CmsShowNavigator : public FWNavigatorBase
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
   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   Int_t realEntry(Int_t rawEntry);
   bool  openFile(const std::string& fileName);
   bool  appendFile(const std::string& fileName, bool checkFileQueueSize, bool live);

   virtual void nextEvent();
   virtual void previousEvent();
   virtual bool nextSelectedEvent();
   virtual bool previousSelectedEvent();
   virtual void firstEvent();
   virtual void lastEvent();
   virtual void goToRunEvent(edm::RunNumber_t, edm::LuminosityBlockNumber_t, edm::EventNumber_t);
   void goTo(FileQueue_i fi, int event);

   void eventFilterEnableCallback(Bool_t);
   void filterEvents();
   void filterEventsAndReset();
  
   void setMaxNumberOfFilesToChain( unsigned int i ) {
      m_maxNumberOfFilesToChain = i;
   }
   
   virtual bool isLastEvent();
   virtual bool isFirstEvent();

   void showEventFilterGUI(const TGWindow* p);
   void applyFiltersFromGUI();
   void toggleFilterEnable();
   void withdrawFilter();
   void resumeFilter();
   
   virtual const edm::EventBase* getCurrentEvent() const;

   const char* frameTitle();
   const char* filterStatusMessage();
   int  getNSelectedEvents();
   int  getNTotalEvents();
   bool canEditFiltersExternally();
   bool filesNeedUpdate() const { return m_filesNeedUpdate; }
   int  getFilterState() { return m_filterState; }

   void editFiltersExternally();

   void activateNewFileOnNextEvent() { m_newFileOnNextEvent = true; }
   void resetNewFileOnNextEvent()    { m_newFileOnNextEvent = false; }

   std::vector<std::string>& getProcessList() const;

   sigc::signal<void, const TFile*> fileChanged_;
   sigc::signal<void> preFiltering_;
   sigc::signal<void, bool> postFiltering_;
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

   void newFile(FileQueue_i);

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
};

#endif
