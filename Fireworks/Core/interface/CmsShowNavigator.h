// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowNavigator_h
#define Fireworks_Core_CmsShowNavigator_h
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.h,v 1.33 2009/11/18 22:46:23 amraktad Exp $
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
private:
   typedef std::deque<FWFileEntry*> FQBase_t;
   typedef FQBase_t::iterator       FQBase_i;


   struct FileQueue_t : public FQBase_t
   {
      struct iterator : public FQBase_i
      {
         bool m_isSet;

         iterator() : m_isSet(false) {}
         iterator(FQBase_i i) : FQBase_i(i), m_isSet(true) {}

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
   std::pair<std::deque<FWFileEntry*>::iterator,Int_t> realEntry(Int_t run, Int_t event);

   bool openFile(const std::string& fileName);
   bool appendFile(const std::string& fileName, bool checkMaxFileSize);

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

   int  getNSelectedEvents();
   int  getNTotalEvents();


   void setMaxNumberOfFilesToChain( unsigned int i ) {
      m_maxNumberOfFilesToChain = i;
   }
   bool isLastEvent();
   bool isFirstEvent();

   void showEventFilterGUI(const TGWindow* p);
   void applyFiltersFromGUI();

   const fwlite::Event* getCurrentEvent() { return (*m_currentFile)->event();}
   
   sigc::signal<void, const fwlite::Event&> newEvent_;
   sigc::signal<void, const TFile*> fileChanged_;
   sigc::signal<void> preFiltering_;
   sigc::signal<void> postFiltering_;
   sigc::signal<void, int, int> eventFilterMessageChanged_;
   sigc::signal<void, bool, bool>updateEventFilterEnable_;
   sigc::signal<void, bool>editFilters_;

private:
   CmsShowNavigator(const CmsShowNavigator&);    // stop default
   const CmsShowNavigator& operator=(const CmsShowNavigator&);    // stop default
   void setCurrentFile(FileQueue_i);
   void updateFileFilters();

   void removeFilter(std::list<FWEventSelector*>::iterator);
   void addFilter(FWEventSelector*);
   void changeFilter(FWEventSelector*);
   void finishEditFilters();
   
   void newFile(FileQueue_i);

   // ---------- member data --------------------------------
   
   std::list<FWEventSelector*>  m_selectors;
   FileQueue_t m_files;
   FileQueue_i m_currentFile;
   int m_currentEvent;

   bool m_filtersEnabled;
   bool m_filterModeOR;
   bool m_filtersNeedUpdate;
   
   unsigned int m_maxNumberOfFilesToChain;
   // entry is an event index nubmer which runs from 0 to
   // #events or #selected_events depending on if we filter
   // events or not
   const CmsShowMain &m_main;
   FWGUIEventFilter*  m_guiFilter;
};

#endif
