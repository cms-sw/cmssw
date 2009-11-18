// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowNavigator_h
#define Fireworks_Core_CmsShowNavigator_h
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.h,v 1.28 2009/11/13 20:58:17 amraktad Exp $
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
   CmsShowNavigator(const CmsShowMain &);
   virtual ~CmsShowNavigator();

   //configuration management interface
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);

   Int_t realEntry(Int_t rawEntry);
   std::pair<std::deque<FWFileEntry>::iterator,Int_t> realEntry(Int_t run, Int_t event);

   bool openFile(const std::string& fileName);
   bool appendFile(const std::string& fileName, bool checkMaxFileSize);
   void nextEvent();
   void previousEvent();
   void firstEvent();
   void firstEventInCurrentFile();
   void lastEvent();
   void lastEventInCurrentFile();
   void enableEventFiltering( Bool_t);
   void filterEvents();
   void filterEventsAndReset();
   void goToEvent(Int_t,Int_t);
   void setMaxNumberOfFilesToChain( unsigned int i ){
      m_maxNumberOfFilesToChain = i;
   }

   bool isLastEvent();
   bool isFirstEvent();

   void showEventFilter(const TGWindow*);
   void applyFilters();
   void applyFiltersFromGUI();

   const fwlite::Event* getCurrentEvent() { return m_currentFile->event();}
   sigc::signal<void, const fwlite::Event&> newEvent_;
   sigc::signal<void, const fwlite::Event&> oldEvent_;
   sigc::signal<void, const TFile*> fileChanged_;
   sigc::signal<void> atBeginning_;
   sigc::signal<void> atEnd_;

   sigc::signal<void> preFiltering_;
   sigc::signal<void> postFiltering_;

   sigc::signal<void, const std::string&> eventSelectionChanged_;

private:
   CmsShowNavigator(const CmsShowNavigator&);    // stop default
   const CmsShowNavigator& operator=(const CmsShowNavigator&);    // stop default

   typedef std::deque<FWFileEntry>            FileQueue_t;
   typedef std::deque<FWFileEntry>::iterator  FileQueue_i;

   void setCurrentFile(FileQueue_i current);
   void filterEvents(FWFileEntry&, int, std::string);
   bool filterEventsWithCustomParser(FWFileEntry& file, int, std::string);

   // ---------- member data --------------------------------
   unsigned int m_maxNumberOfFilesToChain;
   FileQueue_t m_files;
   FileQueue_i m_currentFile;
   std::vector<FWEventSelector*> m_selectors;
   FileQueue_i m_firstSelectedFile;
   FileQueue_i m_lastSelectedFile;
   FileQueue_i m_lastFile;
   bool m_filterEvents;
   bool m_globalOR;
   // entry is an event index nubmer which runs from 0 to
   // #events or #selected_events depending on if we filter
   // events or not
   int m_currentEntry;
   int m_lastEntry;
   const CmsShowMain &m_main;
   FWGUIEventFilter*  m_guiFilter;

};

#endif
