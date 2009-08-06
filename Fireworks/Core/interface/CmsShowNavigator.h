// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowNavigator_h
#define Fireworks_Core_CmsShowNavigator_h
//
// Package:     newVersion
// Class  :     CmsShowNavigator
//
/**\class CmsShowNavigator CmsShowNavigator.h Fireworks/Core/interface/CmsShowNavigator.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Joshua Berger
//         Created:  Tue Jun 10 14:56:34 EDT 2008
// $Id: CmsShowNavigator.h,v 1.15 2009/07/30 04:10:18 dmytro Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>
#include "TFile.h"

// user include files
#include "DataFormats/FWLite/interface/Event.h"

// forward declarations
class TEventList;
class CSGAction;
class CmsShowMain;

namespace edm {
   class EventID;
}

class CmsShowNavigator
{

public:
   CmsShowNavigator(const CmsShowMain &);
   virtual ~CmsShowNavigator();
   //      void startLoop();
   Int_t realEntry(Int_t rawEntry);
   Int_t realEntry(Int_t run, Int_t event);    // -1 means event not found

   bool loadFile(const std::string& fileName);
   void nextEventChangeAlsoChangeFile(const std::string& fileName);
   void checkPosition();
   void nextEvent();
   void previousEvent();
   void firstEvent();
   void lastEvent();
   void filterEventsAndReset(std::string selection);
   void goToRun(Int_t);
   void goToEvent(Int_t);

   bool autoRewind() const {
      return m_loopMode;
   }
   void setAutoRewind( bool mode ) {
      m_loopMode = mode;
   }

   //      void checkBefore();
   //      void checkAfter();
   //      sigc::signal<void, bool> notBegin;
   //      sigc::signal<void, bool> notEnd;
   sigc::signal<void, const fwlite::Event&> newEvent;
   sigc::signal<void, const fwlite::Event&> oldEvent;
   //      sigc::signal<void, int> newEventIndex; //To be replaced when we can get index from fwlite::Event
   sigc::signal<void, const TFile*> newFileLoaded;
   sigc::signal<void> atBeginning;
   sigc::signal<void> atEnd;
   
   sigc::signal<void> preFiltering;
   sigc::signal<void> postFiltering;

private:
   CmsShowNavigator(const CmsShowNavigator&);    // stop default

   const CmsShowNavigator& operator=(const CmsShowNavigator&);    // stop default

   // ---------- member data --------------------------------
   TFile *m_file;
   fwlite::Event *m_event;
   edm::EventID m_firstID;
   edm::EventID m_lastID;
   TTree *m_eventTree;
   std::string m_selection;
   TEventList *m_eventList;
   int m_currentEntry;
   int m_nEntries;
   int m_currentSelectedEntry;
   const CmsShowMain         &m_main;
   bool m_loopMode;    // auto-rewind event loop
   std::string m_nextFile;
};


#endif
