// -*- C++ -*-
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.cc,v 1.36 2009/09/29 19:26:33 dmytro Exp $
//

// hacks
#define private public
#include "DataFormats/FWLite/interface/Event.h"
#include "CmsShowMain.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private

// system include files
#include <string>
#include <boost/regex.hpp>
#include "TTree.h"
#include "TFile.h"
#include "TEventList.h"
#include "TError.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TBranch.h"

// user include files
#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
//
// constructors and destructor
//
CmsShowNavigator::CmsShowNavigator(const CmsShowMain &main)
  : m_maxNumberOfFilesToChain(1),
    m_currentFile(m_files.begin()),
    m_filterEvents(false),
    m_globalOR(true),
    m_currentEntry(0),
    m_lastEntry(-1),
    m_main(main)
{
}

CmsShowNavigator::~CmsShowNavigator()
{
}

//
// member functions
//

bool
CmsShowNavigator::loadFile(const std::string& fileName)
{
   gErrorIgnoreLevel = 3000; // suppress warnings about missing dictionaries
   TFile *newFile = TFile::Open(fileName.c_str());
   if (newFile == 0 || newFile->IsZombie() || !newFile->Get("Events")) {
     std::cout << "Invalid file. Ignored." << std::endl;
     return false;
   }
   gErrorIgnoreLevel = -1;

   //??  CmsShowMain::resetFieldEstimate();
   
   // remove extra files
   if ( m_files.size() > 0 && m_files.size() == m_maxNumberOfFilesToChain ){
     if (m_files.front().file){
       m_files.front().file->Close();
       delete m_files.front().file;
     }
     if (m_files.front().event) delete m_files.front().event;
     m_files.pop_front();
   }

   m_files.push_back(FWFileEntry());
   m_files.back().file = newFile;
   m_files.back().name = fileName;
   fileChanged_.emit(newFile);
   m_files.back().event = new fwlite::Event(newFile);
   m_files.back().eventTree = dynamic_cast<TTree*> (newFile->Get("Events"));
   assert(m_files.back().eventTree!=0 && "Cannot find TTree 'Events' in the data file");

   m_lastFile = m_files.end(); 
   --m_lastFile;
   m_currentFile = m_lastFile;
   filterEvents();
   // firstEventInTheCurrentFile();
   return true;
}

Int_t
CmsShowNavigator::realEntry(Int_t selectedEntry) {
  if (m_filterEvents && m_currentFile != m_files.end())
    return m_currentFile->mainSelection.GetEntry(selectedEntry);
  else
    return selectedEntry;
}

std::pair<std::deque<FWFileEntry>::iterator,Int_t>
CmsShowNavigator::realEntry(Int_t run, Int_t event) {
  for (std::deque<FWFileEntry>::iterator file = m_files.begin();
       file != m_files.end(); ++file)
    {
      file->event->fillFileIndex();
      edm::FileIndex::const_iterator i = 
	file->event->fileIndex_.findEventPosition(run, 0, event, true);
      if (file->event->fileIndex_.end() != i)
	return std::pair<std::deque<FWFileEntry>::iterator,Int_t>(file,i->entry_);
    }
  m_currentFile->file->cd();
  return std::pair<std::deque<FWFileEntry>::iterator,Int_t>(m_files.end(),-1);
}

void
CmsShowNavigator::checkPosition() 
{
  if ( m_filterEvents )
    {
      if (m_currentFile == m_lastSelectedFile &&
	  m_currentEntry == m_currentFile->mainSelection.GetN()-1)
	atEnd_.emit(true);
      else
	atEnd_.emit(false);
    }
  else
    {
      if (m_currentFile == m_lastFile &&
	  m_currentEntry == m_currentFile->eventTree->GetEntries()-1)
	atEnd_.emit(true);
      else
	atEnd_.emit(false);
    }
  if (m_currentEntry == 0){
    if ( m_currentFile == m_files.begin() ||
	 (m_filterEvents && m_currentFile == m_firstSelectedFile) )
      atBeginning_.emit(true);
    else
      atBeginning_.emit(false);
  }
    else
      atBeginning_.emit(false);
}

void
CmsShowNavigator::newRemoteFile(const std::string& fileName)
{
   m_nextFile = fileName;
}

void
CmsShowNavigator::checkForNewFiles()
{
  if( m_nextFile.empty()) return;
  bool loadedNewFile = loadFile(m_nextFile);
  m_nextFile.clear();
  if (loadedNewFile && 
      (!m_filterEvents ||
       m_files.rbegin()->anySelectedEvents() ||
       m_maxNumberOfFilesToChain == 1) )
    {
      m_currentFile = m_lastFile;
      firstEventInTheCurrentFile();
      return;
    }
}

void
CmsShowNavigator::nextEvent()
{
  checkForNewFiles();
  if (m_currentFile == m_files.end()) return;
  
  if ( m_currentEntry == m_lastEntry ){
    ++m_currentFile;
    // first look forward for files with events that pass the selection
    if (m_filterEvents){
      while( m_currentFile != m_files.end() &&
	     !m_currentFile->anySelectedEvents() )
	++m_currentFile;
    }
    // if we reached the end (one way or the other) try from the begging
    if (m_currentFile == m_files.end()) {
      m_currentFile = m_files.begin();
      if (m_filterEvents){
	while( m_currentFile != m_files.end() &&
	       !m_currentFile->anySelectedEvents() )
	  ++m_currentFile;
      }
      assert(m_currentFile != m_files.end() && "fatal file navigation problem"); 
      // this should never happen, because at least one event should have passed 
      // the selection, i.e. the current one.
    } 
    fileChanged_.emit(m_currentFile->file);
    firstEventInTheCurrentFile();
    return;
  }

  if (m_currentEntry < m_lastEntry &&
      m_currentFile->event->to(realEntry(m_currentEntry+1)) ) 
    {
      ++m_currentEntry;
      newEvent_.emit(*(m_currentFile->event));
      checkPosition();
    }
  else 
    oldEvent_.emit(*(m_currentFile->event));
}

void
CmsShowNavigator::previousEvent()
{
  checkForNewFiles();
  if (m_currentFile == m_files.end()) return;

  if ( m_currentEntry == 0 ){
    // no support for reverse loops
    if (m_filterEvents && m_currentFile == m_firstSelectedFile) return; 
    if (!m_filterEvents && m_currentFile == m_files.begin()) return; 
    --m_currentFile;
    if (m_filterEvents){
      while( m_currentFile != m_firstSelectedFile &&
	     !m_currentFile->anySelectedEvents() )
	--m_currentFile;
    }
    fileChanged_.emit(m_currentFile->file);
    lastEventInTheCurrentFile();
    return;
  }

  if (m_currentEntry > 0 &&
      m_currentFile->event->to(realEntry(m_currentEntry-1)) ) 
    {
      --m_currentEntry;
      newEvent_.emit(*(m_currentFile->event));
      checkPosition();
    }
  else 
    oldEvent_.emit(*(m_currentFile->event));
}

void
CmsShowNavigator::firstEvent()
{
  if (m_filterEvents)
    m_currentFile = m_firstSelectedFile;
  else
    m_currentFile = m_files.begin();
  fileChanged_.emit(m_currentFile->file);
  firstEventInTheCurrentFile();
}

void
CmsShowNavigator::firstEventInTheCurrentFile()
{
  if (m_filterEvents && m_currentFile->mainSelection.GetN() == 0){
    m_currentFile = m_firstSelectedFile;
    fileChanged_.emit(m_currentFile->file);
  }
  m_currentEntry = 0;
  if ( m_filterEvents )
    m_lastEntry = m_currentFile->mainSelection.GetN()-1;
  else
    m_lastEntry = m_currentFile->event->size()-1;
  m_currentFile->event->to(realEntry(m_currentEntry));
  newEvent_.emit(*(m_currentFile->event));
  checkPosition();
}


void
CmsShowNavigator::lastEvent()
{
  if (m_filterEvents)
    m_currentFile = m_lastSelectedFile;
  else
    m_currentFile = m_lastFile;
  fileChanged_.emit(m_currentFile->file);
  lastEventInTheCurrentFile();
}

void
CmsShowNavigator::lastEventInTheCurrentFile()
{
  if ( m_filterEvents )
    m_lastEntry = m_currentFile->mainSelection.GetN()-1;
  else
    m_lastEntry = m_currentFile->event->size()-1;
  m_currentEntry = m_lastEntry;
  m_currentFile->event->to(realEntry(m_currentEntry));
  newEvent_.emit(*(m_currentFile->event));
  checkPosition();
}

void
CmsShowNavigator::goToEvent(Int_t run, Int_t event)
{
  std::pair<std::deque<FWFileEntry>::iterator,Int_t> entry = realEntry(run, event);
  if ( entry.first == m_files.end() ) {
    oldEvent_.emit(*m_currentFile->event);
    return;
  }
  if (!m_filterEvents){
    if ( entry.first->event->to(entry.second) ){
      m_currentFile = entry.first;
      m_currentEntry = entry.second;
      m_lastEntry = m_currentFile->event->size()-1;
      newEvent_.emit(*m_currentFile->event);
      checkPosition();
      return;
    } else {
      oldEvent_.emit(*m_currentFile->event);
      return;
    }
  }

  Int_t index = entry.first->mainSelection.GetIndex(entry.second);
  if ( index < 0 ) {
    std::cout << "WARNING: requested event is not among preselected events! " << std::endl;
    oldEvent_.emit(*m_currentFile->event);
    return;
  }

  if ( entry.first->event->to(entry.second) ){
    m_currentFile = entry.first;
    m_currentEntry = index;
    m_lastEntry = m_currentFile->mainSelection.GetN()-1;
    newEvent_.emit(*m_currentFile->event);
    checkPosition();
    return;
  } else {
    oldEvent_.emit(*m_currentFile->event);
    return;
  }
}

void
CmsShowNavigator::filterEvents(FWFileEntry& file, int iSelector, std::string selection)
{
  // parse selection for known Fireworks expressions
  std::string interpretedSelection = selection;
  for (FWEventItemsManager::const_iterator i = m_main.m_eiManager->begin(),
	 end = m_main.m_eiManager->end(); i != end; ++i) 
    {
      if (*i == 0) continue;
      //FIXME: hack to get full branch name filled
      if ( (*i)->m_event == 0 ){
	(*i)->m_event = file.event;
	(*i)->getPrimaryData();
	(*i)->m_event = 0;
      }
      boost::regex re(std::string("\\$") + (*i)->name());
      interpretedSelection = boost::regex_replace(interpretedSelection, re,
						  (*i)->m_fullBranchName + ".obj");
      // printf("selection after applying s/%s/%s/: %s\n",
      //     (std::string("\\$") + (*i)->name()).c_str(),
      //     ((*i)->m_fullBranchName + ".obj").c_str(),
      //     interpretedSelection.c_str());
    }
  file.file->cd();
  file.eventTree->SetEventList(0);

  //since ROOT will leave any TBranches used in the filtering at the last event,               
  // we need to be able to reset them to what fwlite::Event expects them to be                 
  // we do this by holding onto the old buffers and create temporary new ones                  

  TObjArray* branches = file.eventTree->GetListOfBranches();
  std::vector<void*> previousBranchAddresses;
  previousBranchAddresses.reserve(branches->GetEntriesFast());
  {
    std::auto_ptr<TIterator> pIt( branches->MakeIterator());
    while(TObject* branchObj = pIt->Next()) {
      TBranch* b = dynamic_cast<TBranch*> (branchObj);
      if(0!=b) {
	const char * name = b->GetName();
	unsigned int length = strlen(name);
	if(length > 1 && name[length-1]!='.') {
	  //this is not a data branch so we should ignore it                               
	  previousBranchAddresses.push_back(0);
	  continue;
	}
	//std::cout <<" branch '"<<b->GetName()<<"' "<<static_cast<void*>(b->GetAddress())<<std::endl;                                                                                        
	if(0!=b->GetAddress()) {
	  b->SetAddress(0);
	}
	previousBranchAddresses.push_back(b->GetAddress());
      } else {
	previousBranchAddresses.push_back(0);
      }
    }
  }

  file.eventTree->Draw(Form(">>list%d",iSelector),interpretedSelection.c_str());
  //std::cout << Form("File: %s, selection: %s, number of events passed the selection: %d",
  //file.name.c_str(),interpretedSelection.c_str(), file.lists[iSelector]->GetN()) << std::endl;

  //set the old branch buffers                                                                 
  {
    std::auto_ptr<TIterator> pIt( branches->MakeIterator());
    std::vector<void*>::const_iterator itAddress = previousBranchAddresses.begin();
    while(TObject* branchObj = pIt->Next()) {
      TBranch* b = dynamic_cast<TBranch*> (branchObj);
      if(0!=b && 0!=*itAddress) {
	b->SetAddress(*itAddress);
      }
      ++itAddress;
    }
  }

  file.selections[iSelector] = selection;
}

void
CmsShowNavigator::filterEvents()
{
  preFiltering_();
  // Things to do:
  // - make sure we correct number of TEventList's - 1 per selector
  // - re-filter events for each selector if the selector has changed
  //   - if a list needs to be reused and its flagged as not used, ignore it
  // - make a combined list based on active lists
  m_firstSelectedFile = m_files.end();
  m_lastSelectedFile = m_files.end();
  for (std::deque<FWFileEntry>::iterator file = m_files.begin();
       file != m_files.end(); ++file)
    {
      std::vector<unsigned int> updateList;
      for ( unsigned int i=0; i < m_selectors.size(); ++i )
	{
	  const char* name = Form("list%d",i);
	  if ( i==file->lists.size() ) {
	    file->lists.push_back(new TEventList(name,""));
	    file->lists.back()->SetDirectory(file->file);
	    file->selections.push_back(std::string());
	  }
	  if (!m_selectors[i].removed && m_selectors[i].enabled){
	    //std::cout << "i: " << i << "\n\t m_selectors[i].selection: " << m_selectors[i].selection <<
	    //"\n\t file->selections[i]: " << file->selections[i] << std::endl;
	    if ( m_selectors[i].selection != file->selections[i] )
	      updateList.push_back(i);
	  }
	}
      if ( !m_filterEvents ) continue;
      if ( !updateList.empty() ){
	for (std::vector<unsigned int>::const_iterator i = updateList.begin();
	     i != updateList.end(); ++i){
	  filterEvents(*file,*i,m_selectors[*i].selection);
	}
	// hack to get rid of problems with cached events/branches in fwlite::Event. Just make a new one.
	// if ( file->event != 0 ) delete file->event;
	// file->event = new fwlite::Event(file->file);
      }
      // make main selection file
      file->mainSelection.Clear();
      for ( unsigned int i=0; i < m_selectors.size(); ++i )
	if (!m_selectors[i].removed && m_selectors[i].enabled)
	  if ( m_globalOR || file->mainSelection.GetN()==0 )
	    file->mainSelection.Add(file->lists[i]);
	  else
	    file->mainSelection.Intersect(file->lists[i]);
      // std::cout << Form("File: %s, number of events passed OR of all selections: %d",
      // file->name.c_str(),file->mainSelection.GetN()) 
      // << std::endl;
    }

  // events are not filtered
  if ( !m_filterEvents ){
    m_firstSelectedFile = m_files.begin();
    m_lastSelectedFile = m_files.end();
    --m_lastSelectedFile;
    m_currentFile->file->cd();	  
    postFiltering_();
    eventSelectionChanged_.emit(std::string());
    return;
  }

  unsigned int nPassed(0);
  unsigned int nTotal(0);
  m_firstSelectedFile = m_files.end();
  m_lastSelectedFile = m_files.end();
  for (std::deque<FWFileEntry>::iterator file = m_files.begin();
       file != m_files.end(); ++file)
    {
      nPassed += file->mainSelection.GetN();
      nTotal += file->eventTree->GetEntries();
      if (m_firstSelectedFile == m_files.end() && file->mainSelection.GetN()>0 )
	m_firstSelectedFile = file;
      if (file->mainSelection.GetN()>0 )
	m_lastSelectedFile = file;
    }

  // handle exception when no events are selected
  if (nPassed==0){
    m_filterEvents = false;
    m_firstSelectedFile = m_files.begin();
    m_lastSelectedFile = m_files.end();
    --m_lastSelectedFile;
    postFiltering_();
    firstEvent();
    eventSelectionChanged_.emit(std::string());
    return;
  }

  m_currentFile->file->cd();	  
  postFiltering_();
  eventSelectionChanged_.emit(std::string(Form("Events are filtered. %d out of %d events are shown",nPassed,nTotal)));
}
/*
void 
CmsShowNavigator::filterEventsAndReset(){
  filterEvents();
  if (!m_filterEvents)
    return;
  Int_t index = m_currentFile->mainSelection.GetIndex(m_currentEntry);
  if (index >= 0){
    m_currentEntry = 0;
    m_lastEntry = m_currentFile->mainSelection.GetN()-1;
  }
}
*/

void 
CmsShowNavigator::enableEventFiltering( Bool_t flag ){
  if (flag == m_filterEvents) return;
  m_filterEvents = flag;
  filterEvents();
  if (flag != m_filterEvents) return; // bad case and it's handled already
  if (m_filterEvents){
    // events were not filtered before, check if we can keep the same event open
    Int_t index = m_currentFile->mainSelection.GetIndex(m_currentEntry);
    if (index >= 0){
      m_currentEntry = index;
      m_lastEntry = m_currentFile->mainSelection.GetN()-1;
      checkPosition();
    } else {
      firstEventInTheCurrentFile();
    }
  } else {
    // events were filted before, so now we need just set boundaries and position right
    m_currentEntry = m_currentFile->mainSelection.GetEntry(m_currentEntry);
    m_lastEntry = m_currentFile->event->size()-1;
    checkPosition();
  }
}

void
CmsShowNavigator::showEventFilter(){
  FWGUIEventFilter* filter = new FWGUIEventFilter(m_selectors,m_globalOR);
  filter->show();
  gClient->WaitForUnmap(filter);
  Int_t absolutePosition = m_currentEntry;
  if (m_filterEvents) absolutePosition = m_currentFile->mainSelection.GetEntry(m_currentEntry);
  filterEvents();
  if (!m_filterEvents) return;
  // events were filtered before and filtered now
  // check if we can keep the same event open
  Int_t index = m_currentFile->mainSelection.GetIndex(absolutePosition);
  if (index >= 0){
    m_currentEntry = index;
    m_lastEntry = m_currentFile->mainSelection.GetN()-1;
    checkPosition();
  } else {
    firstEventInTheCurrentFile();
  }
}



void
CmsShowNavigator::setFrom(const FWConfiguration& iFrom) {
  int numberOfFilters(0);
  {  
    const FWConfiguration* value = iFrom.valueForKey( "EventFilter_total" );
    if (!value) return;
    std::istringstream s(value->value());
    s>>numberOfFilters;
  }
  m_selectors.clear();
  {  
    const FWConfiguration* value = iFrom.valueForKey( "EventFilter_enabled" );
    assert(value);
    std::istringstream s(value->value());
    s>>m_filterEvents;
  }
  
  for(int i=0; i<numberOfFilters; ++i){
    FWEventSelector selector;
    {
      const FWConfiguration* value = 
	iFrom.valueForKey( Form("EventFilter%d_enabled",i) );
      assert(value);
      std::istringstream s(value->value());
      s>>selector.enabled;
    }
    {
      const FWConfiguration* value = 
	iFrom.valueForKey( Form("EventFilter%d_selection",i) );
      assert(value);
      std::istringstream s(value->value());
      s>>selector.selection;
    }
    {
      const FWConfiguration* value = 
	iFrom.valueForKey( Form("EventFilter%d_comment",i) );
      assert(value);
      std::istringstream s(value->value());
      s>>selector.title;
    }
    m_selectors.push_back(selector);
  }
}

void
CmsShowNavigator::addTo(FWConfiguration& iTo) const
{
  int numberOfFilters(0);
  for (std::vector<FWEventSelector>::const_iterator sel = m_selectors.begin();
       sel != m_selectors.end(); ++sel){
    if ( sel->removed ) continue;
    iTo.addKeyValue(Form("EventFilter%d_enabled",numberOfFilters),
		    FWConfiguration(Form("%d",sel->enabled)));
    iTo.addKeyValue(Form("EventFilter%d_selection",numberOfFilters),
		    FWConfiguration(sel->selection));
    iTo.addKeyValue(Form("EventFilter%d_comment",numberOfFilters),
		    FWConfiguration(sel->title));
    ++numberOfFilters;
  }
  iTo.addKeyValue("EventFilter_total",FWConfiguration(Form("%d",numberOfFilters)));
  iTo.addKeyValue("EventFilter_enabled",FWConfiguration(Form("%d",m_filterEvents)));
}
