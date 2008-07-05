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
// $Id: CmsShowNavigator.cc,v 1.5 2008/07/05 20:24:30 dmytro Exp $
//

// hacks
#define private public
#include "DataFormats/FWLite/interface/Event.h"
#undef private

// system include files
#include "TTree.h"
#include "TEventList.h"
#include "TError.h"

// user include files
#include "Fireworks/Core/interface/CmsShowNavigator.h"
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
CmsShowNavigator::CmsShowNavigator()
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
CmsShowNavigator::loadFile(std::string fileName) 
{
  if (fileName == "") fileName = "data.root";
  if (m_file != 0) {
    delete m_eventList;
    delete m_eventTree;
    delete m_event;
    m_file->Close();
    delete m_file;
  }
  gErrorIgnoreLevel = 3000; // suppress warnings about missing dictionaries
  TFile *newFile = TFile::Open(fileName.c_str());
  if (newFile == 0) {
    // Throw an exception
    printf("Invalid file\n");
  }
  gErrorIgnoreLevel = -1;
  m_file = newFile;
  m_event = new fwlite::Event(m_file);
  m_eventTree = (TTree*)m_file->Get("Events");
  m_eventList = new TEventList("list","");
  if (m_selection && m_eventTree) {
    m_eventTree->Draw(">>list",m_selection);
    m_eventTree->SetEventList( m_eventList );
  }
  m_nEntries = m_event->size();
  if ( m_eventTree && m_eventTree->GetEventList() ) m_nEntries = m_eventList->GetN();
  newFileLoaded.emit();
  m_event->to(realEntry(0));
  m_firstID = m_event->id();
  m_event->to(realEntry(m_nEntries - 1));
  m_lastID = m_event->id();
  firstEvent();
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
  if (m_currentSelectedEntry < m_nEntries-1 &&
      m_event->to(m_currentSelectedEntry+1) ) {
     ++m_currentSelectedEntry;
     newEvent.emit(*m_event);
     checkPosition();
  }
  else oldEvent.emit(*m_event);
}

void
CmsShowNavigator::previousEvent()
{
  if (m_currentSelectedEntry > 0 &&
      m_event->to(m_currentSelectedEntry-1) ) {
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
CmsShowNavigator::goToRun(Double_t run)
{
   Int_t entry = realEntry(static_cast<UInt_t>(run), 0);
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
CmsShowNavigator::goToEvent(Double_t event)
{
   Int_t entry = realEntry(m_event->id().run(), static_cast<UInt_t>(event));
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

//
// const member functions
//

//
// static member functions
//
