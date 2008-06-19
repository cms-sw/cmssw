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
// $Id: CmsShowNavigator.cc,v 1.1 2008/06/17 00:08:11 chrjones Exp $
//

// system include files
#include "TTree.h"
#include "TEventList.h"
#include "TError.h"

// user include files
#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "DataFormats/FWLite/interface/Event.h"


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
  printf("File name: %s\n", fileName.c_str());
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
  firstEvent();
}

Int_t
CmsShowNavigator::realEntry(Int_t rawEntry) {
  if (m_eventTree && m_eventTree->GetEventList() )
    return m_eventTree->GetEntryNumber(rawEntry);
  else
    return rawEntry;
}

void
CmsShowNavigator::checkPosition() {
  if ( m_currentEntry == 0 )
    atBeginning.emit();
  if ( m_currentEntry == m_nEntries - 1)
    atEnd.emit();
}      
    
void
CmsShowNavigator::nextEvent() 
{
  if ( m_currentEntry == m_nEntries - 1 ) {
    printf("At end of file\n"); //Make throw exception
  }
  else {
    ++m_currentEntry;
    m_event->to(realEntry(m_currentEntry));
    newEvent.emit(*m_event);
    newEventIndex.emit(m_currentEntry);
    checkPosition();
  }
}

void
CmsShowNavigator::previousEvent()
{
  if ( m_currentEntry == 0 ) {
    printf("At begining of file\n"); //Make throw exception
  }
  else {
    --m_currentEntry;
    m_event->to(realEntry(m_currentEntry));
    newEvent.emit(*m_event);
    newEventIndex.emit(m_currentEntry);
    checkPosition();
  }
}

void
CmsShowNavigator::firstEvent()
{
  m_currentEntry = 0;
  m_event->to(realEntry(m_currentEntry));
  newEvent.emit(*m_event);
  newEventIndex.emit(m_currentEntry);
  atBeginning.emit();
}

void
CmsShowNavigator::event(Int_t i)
{
  if (i > m_nEntries || i < 0) {
    printf("Invalid selection: larger than number of entries\n");//Throw exception
  }
  else {
    m_currentEntry = i;
    m_event->to(realEntry(m_currentEntry));
  }
  newEvent.emit(*m_event);
  newEventIndex.emit(m_currentEntry);
  checkPosition();
}

//
// const member functions
//

//
// static member functions
//
