// -*- C++ -*-
//
// Package:     newVersion
// Class  :     CmsShowNavigator
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue Jun 10 14:56:46 EDT 2008
// $Id: CmsShowNavigator.cc,v 1.16 2008/08/30 18:33:09 dmytro Exp $
//

// hacks
#define private public
#include "DataFormats/FWLite/interface/Event.h"
#undef private

// system include files
#include <string>
#include <boost/regex.hpp>
#include "TTree.h"
#include "TEventList.h"
#include "TError.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
// user include files
#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#define private public
#include "CmsShowMain.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Provenance/interface/EventID.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowNavigator::CmsShowNavigator(const CmsShowMain &main)
     : m_main(main),
 m_loopMode(false)
{
  m_file = 0;
  m_eventTree = 0;
  m_eventList = 0;
  m_selection = "";
  m_currentSelectedEntry = 0;
}

// CmsShowNavigator::CmsShowNavigator(const CmsShowNavigator& rhs)
// {
//    // do actual copying here;
// }

CmsShowNavigator::~CmsShowNavigator()
{
}

//
// assignment operators
//
// const CmsShowNavigator& CmsShowNavigator::operator=(const CmsShowNavigator& rhs)
// {
//   //An exception safe implementation is
//   CmsShowNavigator temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CmsShowNavigator::loadFile(const std::string& fileName)
{
   gErrorIgnoreLevel = 3000; // suppress warnings about missing dictionaries
   TFile *newFile = TFile::Open(fileName.c_str());
   if (newFile == 0) {
      // Throw an exception
      printf("Invalid file\n");
      return;
   }
   if (m_file != 0) {
      delete m_eventList;
      delete m_eventTree;
      delete m_event;
      m_file->Close();
      delete m_file;
   }

   gErrorIgnoreLevel = -1;
   m_file = newFile;
   newFileLoaded.emit(m_file);
   m_event = new fwlite::Event(m_file);
   m_eventTree = dynamic_cast<TTree*> (m_file->Get("Events"));
   assert(m_eventTree!=0);
   m_eventList = new TEventList("list","");
   filterEventsAndReset(m_selection.c_str()); // first event is loaded at the end
}

void
CmsShowNavigator::nextEventChangeAlsoChangeFile(const std::string& fileName)
{
   m_nextFile = fileName;
}


Int_t
CmsShowNavigator::realEntry(Int_t selectedEntry) {
  if (m_eventTree && m_eventTree->GetEventList() )
    return m_eventTree->GetEntryNumber(selectedEntry);
  else
    return selectedEntry;
}

Int_t
CmsShowNavigator::realEntry(Int_t run, Int_t event) {
   m_event->fillFileIndex();
   edm::FileIndex::const_iterator i = m_event->fileIndex_.findEventPosition(run, 0, event, true);
   if (m_event->fileIndex_.end() != i)
     return i->entry_;
   else
     return -1;
}

void
CmsShowNavigator::checkPosition() {
  if ( m_event->id() == m_firstID )
    atBeginning.emit();
  if ( m_event->id() == m_lastID)
    atEnd.emit();
}

void
CmsShowNavigator::nextEvent()
{
   if( ! m_nextFile.empty()) {
      loadFile(m_nextFile);
      m_nextFile.clear();
      return;
   }
   if ( m_loopMode &&
       m_currentSelectedEntry == m_nEntries-1 ) {
      firstEvent();
      return;
   }
   if (m_currentSelectedEntry < m_nEntries-1 &&
      m_event->to(realEntry(m_currentSelectedEntry+1)) ) {
     ++m_currentSelectedEntry;
     newEvent.emit(*m_event);
     checkPosition();
   } else {
      oldEvent.emit(*m_event);
   }
}

void
CmsShowNavigator::previousEvent()
{
   if( ! m_nextFile.empty()) {
      loadFile(m_nextFile);
      m_nextFile.clear();
      return;
   }
   if (m_currentSelectedEntry > 0 &&
      m_event->to(realEntry(m_currentSelectedEntry-1)) ) {
     --m_currentSelectedEntry;
     newEvent.emit(*m_event);
     checkPosition();
  }
  else oldEvent.emit(*m_event);
}

void
CmsShowNavigator::firstEvent()
{
  m_currentSelectedEntry = 0;
  m_event->to(realEntry(m_currentSelectedEntry));
  newEvent.emit(*m_event);
  atBeginning.emit();
}

void
CmsShowNavigator::goToRun(CSGAction* action)
{
   Long_t run = action->getNumberEntry()->GetIntNumber();
   Int_t entry = realEntry(run, 0);
   if ( entry < 0 ) {
      oldEvent.emit(*m_event);
      return;
   }
   Int_t index = entry;
   if (m_eventTree && m_eventTree->GetEventList() ) index = m_eventTree->GetEventList()->GetIndex(entry);
   if (m_event->to(entry)) {
      if ( index < 0 )
	std::cout << "WARNING: requested event is not among preselected events! " << std::endl;
      else
	m_currentSelectedEntry = index;
      newEvent.emit(*m_event);
      checkPosition();
   }
   else oldEvent.emit(*m_event);
}

void
CmsShowNavigator::goToEvent(CSGAction* action)
{
   Long_t event = action->getNumberEntry()->GetIntNumber();
   Int_t entry = realEntry(m_event->id().run(), event);
   if ( entry < 0 ) {
      oldEvent.emit(*m_event);
      return;
   }
   Int_t index = entry;
   if (m_eventTree && m_eventTree->GetEventList() ) index = m_eventTree->GetEventList()->GetIndex(entry);
   if (m_event->to(entry)) {
      if ( index < 0 )
	std::cout << "WARNING: requested event is not among preselected events! " << std::endl;
      else
	m_currentSelectedEntry = index;
      newEvent.emit(*m_event);
      checkPosition();
   }
   else oldEvent.emit(*m_event);
}

void
CmsShowNavigator::filterEvents(CSGAction* action)
{
   if ( action->getTextEntry() )
     filterEventsAndReset( action->getTextEntry()->GetText() );
}

void
CmsShowNavigator::filterEventsAndReset(const char* sel)
{
     std::string selection = sel;
     for (FWEventItemsManager::const_iterator i = m_main.m_eiManager->begin(),
	       end = m_main.m_eiManager->end();
	  i != end;
	  ++i) {
	  if (*i == 0)
	       continue;
	  boost::regex re(std::string("\\$") + (*i)->name());
	  std::string new_sel =
	       boost::regex_replace(selection, re,
				    (*i)->m_fullBranchName + ".obj");
// 	  printf("selection after applying s/%s/%s/: %s\n",
// 		 (std::string("\\$") + (*i)->name()).c_str(),
// 		 (*i)->moduleLabel().c_str(),
// 		 new_sel.c_str());
	  selection.swap(new_sel);
     }
//      std::string s = selection;
//      for (boost::sregex_iterator i = boost::sregex_iterator(s.begin(), s.end(), re),
// 	       end;
// 	  i != end;
// 	  ++i) {
// 	  printf("%s\n", i->str(0).c_str());
//      }
//      return;
     m_selection = selection;
     m_eventTree->SetEventList(0);
     if ( m_selection.length() != 0 ) {
// 	  std::cout << "Selection requested: " << m_selection << std::endl;
	  m_eventTree->Draw(">>list",m_selection.c_str());
	  m_eventTree->SetEventList( m_eventList );
     }
     m_nEntries = m_event->size();
     if ( m_eventTree->GetEventList() ){
	m_nEntries = m_eventList->GetN();
	if ( m_nEntries < 1 ) {
	   std::cout << "WARNING: No events passed selection: " << sel << std::endl;
	   m_eventTree->SetEventList(0);
	   m_nEntries = m_event->size();
	}
     }
     m_event->to(realEntry(0));
     m_firstID = m_event->id();
     m_event->to(realEntry(m_nEntries - 1));
     m_lastID = m_event->id();
     firstEvent();
}


//
// const member functions
//

//
// static member functions
//
