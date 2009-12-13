// -*- C++ -*-
#ifndef Fireworks_Core_FWFileEntry_h
#define Fireworks_Core_FWFileEntryr_h
//
// Package:     Core
// Class  :     FWFileEntry
// $Id: CmsShowNavigator.h,v 1.25 2009/10/27 01:55:28 dmytro Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "TEventList.h"

// forward declarations
class TEventList;
class CSGAction;
class CmsShowMain;
class TFile;
class TGWindow;

namespace edm {
   class EventID;
}

class FWFileEntry {
public:
   FWFileEntry(const std::string& name);
   bool anySelectedEvents() const {
      if ( m_eventTree && m_mainSelection.GetN()>0 )
         return true;
      else
         return false;
   }
   TFile*         file(){
      return m_file;
   }
   fwlite::Event* event(){
      return m_event;
   }
   TTree*         tree(){
      return m_eventTree;
   }
   TEventList&    mainSelection(){
      return m_mainSelection;
   }
   std::vector<std::string>& selections(){
      return m_selections;
   }
   std::vector<TEventList*>& lists(){
      return m_lists;
   }
   bool openFile();
   void closeFile();
private:
   // file name
   std::string m_name;
   // named lists in order to use in Draw() command living in the file directory
   std::vector<TEventList*> m_lists;
   std::vector<std::string> m_selections;
   // unnamed main selection list
   TEventList m_mainSelection;
   TFile* m_file;
   TTree* m_eventTree;
   fwlite::Event* m_event;
};
#endif
