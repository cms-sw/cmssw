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
// $Id: CmsShowNavigator.h,v 1.18 2009/08/12 12:49:44 amraktad Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

// user include files
#include "DataFormats/FWLite/interface/Event.h"

// forward declarations
class TEventList;
class CSGAction;
class CmsShowMain;
class TFile;

namespace edm {
   class EventID;
}

class CmsShowNavigator
{
public:
   CmsShowNavigator(const CmsShowMain &);
   virtual ~CmsShowNavigator();

   Int_t realEntry(Int_t rawEntry);
   Int_t realEntry(Int_t run, Int_t event);    // -1 means event not found

   bool loadFile(const std::string& fileName);
   void nextEventChangeAlsoChangeFile(const std::string& fileName);
   void checkPositionInGoTo();
   void nextEvent();
   void previousEvent();
   void firstEvent();
   void lastEvent();
   void filterEventsAndReset(std::string selection);
   void goToRun(Int_t);
   void goToEvent(Int_t);

   sigc::signal<void, const fwlite::Event&> newEvent_;
   sigc::signal<void, const fwlite::Event&> oldEvent_;
   sigc::signal<void, const TFile*> newFileLoaded_;
   sigc::signal<void> atBeginning_;
   sigc::signal<void> atEnd_;

   sigc::signal<void> preFiltering_;
   sigc::signal<void> postFiltering_;

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
   int m_lastEntry;
   int m_currentSelectedEntry;
   const CmsShowMain         &m_main;
   std::string m_nextFile;
};

#endif
