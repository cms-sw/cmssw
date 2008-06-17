#ifndef Fireworks_Core_CmsShowNavigator_h
#define Fireworks_Core_CmsShowNavigator_h
// -*- C++ -*-
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
// $Id$
//

// system include files
#include <sigc++/sigc++.h>
#include "TFile.h"

// user include files
#include "DataFormats/FWLite/interface/Event.h"

// forward declarations
class TEventList;

class CmsShowNavigator
{

   public:
      CmsShowNavigator();
      virtual ~CmsShowNavigator();
      //      void startLoop();
      Int_t realEntry(Int_t rawEntry);
      // ---------- const member functions --------------------- 
      void loadFile(std::string fileName);
      void checkPosition();
      void nextEvent();
      void previousEvent();
      void firstEvent();
      void event(Int_t i);
      // ---------- static member functions --------------------

      // ---------- member functions --------------------------- 
      //      void checkBefore();
      //      void checkAfter();
      //      sigc::signal<void, bool> notBegin;
      //      sigc::signal<void, bool> notEnd;
      sigc::signal<void, const fwlite::Event&> newEvent;
      sigc::signal<void, int> newEventIndex; //To be replaced when we can get index from fwlite::Event
      sigc::signal<void> newFileLoaded;
      sigc::signal<void> atBeginning;
      sigc::signal<void> atEnd;

   private:
      CmsShowNavigator(const CmsShowNavigator&); // stop default

      const CmsShowNavigator& operator=(const CmsShowNavigator&); // stop default

      // ---------- member data --------------------------------
      TFile *m_file;
      fwlite::Event *m_event;
      TTree *m_eventTree;
      const char *m_selection;
      TEventList *m_eventList;
      int m_currentEntry;
      int m_nEntries;
};


#endif
