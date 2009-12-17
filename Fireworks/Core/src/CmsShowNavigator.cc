// -*- C++ -*-
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.cc,v 1.46 2009/11/14 11:31:13 amraktad Exp $
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
#include "TROOT.h"
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
#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

//
// constructors and destructor
//
CmsShowNavigator::CmsShowNavigator(const CmsShowMain &main):
   m_maxNumberOfFilesToChain(1),
   m_currentFile(m_files.begin()),
   m_filterEvents(false),
   m_globalOR(true),
   m_currentEntry(0),
   m_lastEntry(-1),
   m_main(main),
   m_guiFilter(0)
{
}

CmsShowNavigator::~CmsShowNavigator()
{
}

//
// member functions
//

bool
CmsShowNavigator::openFile(const std::string& fileName)
{
   while ( m_files.size() > 0 )
   {
      m_files.front().closeFile();
      m_files.pop_front();
   }
   FWFileEntry newFile(fileName);
   m_files.push_back(newFile);
   filterEvents();
   
   m_lastFile = m_files.end();
   --m_lastFile;
   
   return true;
}

bool
CmsShowNavigator::appendFile(const std::string& fileName, bool checkMaxFileSize)
{
   FWFileEntry newFile(fileName);
   if ( newFile.file() == 0 ) return false; //bad file

    unsigned int nFilesKeep = checkMaxFileSize ? m_maxNumberOfFilesToChain : 1;
   // remove extra files
   while ( m_files.size() > 0 && m_files.size() >= nFilesKeep &&
           m_files.begin() != m_currentFile)
   {
      m_files.front().closeFile();
      m_files.pop_front();
   }
   
   if (m_files.size() >= m_maxNumberOfFilesToChain)
      printf("WARNING:: %d chained files more than maxNumberOfFilesToChain [%d]\n", (int)m_files.size(), m_maxNumberOfFilesToChain);
   
   m_files.push_back(newFile);
   m_lastFile = m_files.end();
   --m_lastFile;

   filterEvents();

   return true;
}

//______________________________________________________________________________

Int_t
CmsShowNavigator::realEntry(Int_t entry)
{
   if (m_filterEvents && m_currentFile != m_files.end())
      return m_currentFile->mainSelection().GetEntry(entry);
   else
      return entry;
}

std::pair<CmsShowNavigator::FileQueue_i,Int_t>
CmsShowNavigator::realEntry(Int_t run, Int_t event)
{
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      file->event()->fillFileIndex();
      edm::FileIndex::const_iterator i =
         file->event()->fileIndex_.findEventPosition(run, 0, event, true);
      if (file->event()->fileIndex_.end() != i)
         return std::pair<FileQueue_i,Int_t>(file,i->entry_);
   }

   return std::pair<FileQueue_i,Int_t>(m_files.end(),-1);
}

//______________________________________________________________________________
void
CmsShowNavigator::setCurrentFile(FileQueue_i fi)
{
   m_currentFile = fi;
   if (m_filterEvents)
   {
      m_currentFile = m_lastSelectedFile;
      m_lastEntry   = m_currentFile->mainSelection().GetN()-1;
   }
   else
   {
      m_currentFile = m_lastFile;
      m_lastEntry   = m_currentFile->event()->size()-1;
   }
   fileChanged_.emit(m_currentFile->file());
}

void
CmsShowNavigator::firstEvent()
{
   setCurrentFile(m_lastFile);
   firstEventInCurrentFile();
}

void
CmsShowNavigator::firstEventInCurrentFile()
{
   m_currentEntry = 0;
   Int_t re = realEntry(m_currentEntry);
   m_currentFile->event()->to(re);
   newEvent_.emit(*(m_currentFile->event()));
}

void
CmsShowNavigator::lastEvent()
{
   setCurrentFile(m_lastFile);
   lastEventInCurrentFile();
}

void
CmsShowNavigator::lastEventInCurrentFile()
{
   m_currentEntry = m_lastEntry;
   m_currentFile->event()->to(realEntry(m_currentEntry));
   newEvent_.emit(*(m_currentFile->event()));
}

//______________________________________________________________________________
void
CmsShowNavigator::nextEvent()
{
   if (m_currentFile == m_files.end()) return;

   if ( m_currentEntry == m_lastEntry )
   {
      FileQueue_i fi = m_currentFile;
      ++fi;
      if (m_filterEvents) {
         while( fi != m_files.end() && !fi->anySelectedEvents() )
            ++m_currentFile;
      }
      if (fi == m_files.end()) return;

      setCurrentFile(fi);
      firstEventInCurrentFile();
   }
   else if (m_currentEntry < m_lastEntry &&
            m_currentFile->event()->to(realEntry(m_currentEntry+1)) )
   {
      ++m_currentEntry;
      newEvent_.emit(*(m_currentFile->event()));
   }
}

void
CmsShowNavigator::previousEvent()
{
   if (m_currentFile == m_files.end()) return;

   if ( m_currentEntry == 0 )
   {
      // no support for reverse loops
      if (m_filterEvents && m_currentFile == m_firstSelectedFile) return;
      if (!m_filterEvents && m_currentFile == m_files.begin()) return;
      --m_currentFile;
      if (m_filterEvents) {
         while( m_currentFile != m_firstSelectedFile &&
                !m_currentFile->anySelectedEvents() )
            --m_currentFile;
      }
      fileChanged_.emit(m_currentFile->file());
      lastEventInCurrentFile();
      return;
   }

   if (m_currentEntry > 0 &&
       m_currentFile->event()->to(realEntry(m_currentEntry-1)) )
   {
      --m_currentEntry;
      newEvent_.emit(*(m_currentFile->event()));
   }
   else
   {
      oldEvent_.emit(*(m_currentFile->event()));
   }
}

void
CmsShowNavigator::goToEvent(Int_t run, Int_t event)
{
   bool isNew = false;
   std::pair<std::deque<FWFileEntry>::iterator,Int_t> entry = realEntry(run, event);

   if ( entry.first != m_files.end() )
   {
      if (m_filterEvents)
      {
         Int_t index = entry.first->mainSelection().GetIndex(entry.second);
         if ( index > 0 )
         {
            if (m_currentFile  != entry.first ) setCurrentFile(entry.first);
            entry.first->event()->to(entry.second);
            m_currentEntry = index;
            isNew = true;
         }
         else {
            std::cout << "WARNING: requested event is not among preselected events! " << std::endl;
         }
      }
      else
      {
         if ( entry.first->event()->to(entry.second) ) {
         if (m_currentFile  != entry.first ) setCurrentFile(entry.first);
            m_currentEntry = entry.second;
            isNew = true;
         }
      }
   }

   isNew ? oldEvent_.emit(*m_currentFile->event()) : oldEvent_.emit(*m_currentFile->event());
   return;
}

void
CmsShowNavigator::applyFilters()
{
   filterEvents();
   bool oldEntryMatch = false;
   std::pair<std::deque<FWFileEntry>::iterator,Int_t> entry = realEntry(m_currentFile->event()->id().run(), m_currentEntry);
   if ( entry.first != m_files.end() )
   {
      if (m_filterEvents)
      {
         Int_t index = entry.first->mainSelection().GetIndex(entry.second);
         if ( index > 0  && m_currentFile  != entry.first &&  entry.first->event()->to(entry.second))
         {
            setCurrentFile(entry.first);
            m_currentEntry = index;
            oldEntryMatch = true;
         }
      }
      else
      {
         if ( m_currentFile  != entry.first && entry.first->event()->to(entry.second) ) {
            setCurrentFile(entry.first);
            m_currentEntry = entry.second;
            oldEntryMatch = true;
         }
      }
   }

   if (!oldEntryMatch) firstEvent();
   return;
}

void
CmsShowNavigator::applyFiltersFromGUI()
{
   m_globalOR = m_guiFilter->isLogicalOR();
   applyFilters();
}

void
CmsShowNavigator::enableEventFiltering(Bool_t enabled)
{
   if (enabled == m_filterEvents)
      return;

   m_filterEvents = !m_filterEvents;
   applyFilters();
}


void
CmsShowNavigator::filterEvents(FWFileEntry& file, int iSelector, std::string selection)
{
   file.selections()[iSelector] = selection;
   // first our parser
   if (filterEventsWithCustomParser(file, iSelector, selection)) return;

   // parse selection for known Fireworks expressions
   std::string interpretedSelection = selection;
   for (FWEventItemsManager::const_iterator i = m_main.m_eiManager->begin(),
        end = m_main.m_eiManager->end(); i != end; ++i)
   {
      if (*i == 0) continue;
      //FIXME: hack to get full branch name filled
      if ( (*i)->m_event == 0 ) {
         (*i)->m_event = file.event();
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
   file.file()->cd();
   file.tree()->SetEventList(0);

   //since ROOT will leave any TBranches used in the filtering at the last event,
   // we need to be able to reset them to what fwlite::Event expects them to be
   // we do this by holding onto the old buffers and create temporary new ones

   TObjArray* branches = file.tree()->GetListOfBranches();
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

   Int_t result = file.tree()->Draw(Form(">>list%d",iSelector),interpretedSelection.c_str());
   //std::cout << Form("File: %s, selection: %s, number of events passed the selection: %d",
   //file.name.c_str(),interpretedSelection.c_str(), file.lists[iSelector]->GetN()) << std::endl;

   if (result<0) {
      std::cout << "Selection: \"" << selection << "\" is invalid. Disabled." <<std::endl;
      m_selectors[iSelector]->enabled = false;
   }

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
   m_lastSelectedFile  = m_files.end();
   for (std::deque<FWFileEntry>::iterator file = m_files.begin();
        file != m_files.end(); ++file)
   {
      std::vector<unsigned int> updateList;
      for ( unsigned int i=0; i < m_selectors.size(); ++i )
      {
         const char* name = Form("list%d",i);
         if ( i==file->lists().size() ) {
            file->lists().push_back(new TEventList(name,""));
            file->lists().back()->SetDirectory(file->file());
            file->selections().push_back(std::string());
         }
         if (!m_selectors[i]->removed && m_selectors[i]->enabled) {
            //std::cout << "i: " << i << "\n\t m_selectors[i]->selection: " << m_selectors[i]->selection <<
            //"\n\t file->selections[i]: " << file->selections[i] << std::endl;
            if ( m_selectors[i]->selection != file->selections()[i] )
               updateList.push_back(i);
         }
      }
      if ( !m_filterEvents ) continue;
      if ( !updateList.empty() ) {
         for (std::vector<unsigned int>::const_iterator i = updateList.begin();
              i != updateList.end(); ++i) {
            filterEvents(*file,*i,m_selectors[*i]->selection);
         }
         // hack to get rid of problems with cached events/branches in fwlite::Event. Just make a new one.
         // if ( file->event != 0 ) delete file->event;
         // file->event = new fwlite::Event(file->file);
      }
      // make main selection file
      file->mainSelection().Clear();
      for ( unsigned int i=0; i < m_selectors.size(); ++i )
         if (!m_selectors[i]->removed && m_selectors[i]->enabled)
            if ( m_globalOR || file->mainSelection().GetN()==0 )
               file->mainSelection().Add(file->lists()[i]);
            else
               file->mainSelection().Intersect(file->lists()[i]);
      // std::cout << Form("File: %s, number of events passed OR of all selections: %d",
      // file->name.c_str(),file->mainSelection.GetN())
      // << std::endl;
   }

   // events are not filtered
   if ( !m_filterEvents ) {
      m_firstSelectedFile = m_files.begin();
      m_lastSelectedFile  = m_files.end();
      --m_lastSelectedFile;
      postFiltering_();
      eventSelectionChanged_.emit(std::string());
      return;
   }

   unsigned int nPassed(0);
   unsigned int nTotal (0);
   m_firstSelectedFile = m_files.end();
   m_lastSelectedFile  = m_files.end();
   for (std::deque<FWFileEntry>::iterator file = m_files.begin();
        file != m_files.end(); ++file)
   {
      nPassed += file->mainSelection().GetN();
      nTotal  += file->tree()->GetEntries();
      if (m_firstSelectedFile == m_files.end() && file->mainSelection().GetN()>0 )
         m_firstSelectedFile = file;
      if (file->mainSelection().GetN()>0 )
         m_lastSelectedFile = file;
   }

   // handle exception when no events are selected
   if (nPassed == 0) {
      m_filterEvents = false;
      m_firstSelectedFile = m_files.begin();
      m_lastSelectedFile  = m_files.end();
      --m_lastSelectedFile;
      postFiltering_();
      eventSelectionChanged_.emit(std::string());
      return;
   }

   postFiltering_();
   eventSelectionChanged_.emit(std::string(Form("Events are filtered. %d out of %d events are shown",nPassed,nTotal)));

   gROOT->cd();
}


bool
CmsShowNavigator::filterEventsWithCustomParser(FWFileEntry& file, int iSelector, std::string selection)
{
   // get rid of white spaces
   boost::regex re_spaces("\\s+");
   selection = boost::regex_replace(selection,re_spaces,"");
   edm::EventID currentEvent = file.event()->id();
   fwlite::Handle<edm::TriggerResults> hTriggerResults;
   fwlite::TriggerNames const* triggerNames(0);
   try
   {
      hTriggerResults.getByLabel(*file.event(),"TriggerResults","","HLT");
      triggerNames = &file.event()->triggerNames(*hTriggerResults);
   }
   catch(...)
   {
      std::cout << "Warning: failed to get trigger results with process name HLT" << std::endl;
      return false;
   }

   // std::cout << "Number of trigger names: " << triggerNames->size() << std::endl;
   // for (unsigned int i=0; i<triggerNames->size(); ++i)
   //  std::cout << " " << triggerNames->triggerName(i);
   //std::cout << std::endl;

   // cannot interpret selection with OR and AND
   if (selection.find("&&")!=std::string::npos &&
       selection.find("||")!=std::string::npos )
   {
      return false;
   }

   bool junction_mode = true; // AND
   if (selection.find("||")!=std::string::npos)
      junction_mode = false; // OR

   boost::regex re("\\&\\&|\\|\\|");
   boost::sregex_token_iterator i(selection.begin(), selection.end(), re, -1);
   boost::sregex_token_iterator j;

   // filters and how they enter in the logical expression
   std::vector<std::pair<unsigned int,bool> > filters;

   while(i != j)
   {
      std::string filter = *i++;
      bool flag = true;
      if (filter[0]=='!') {
         flag = false;
         filter.erase(filter.begin());
      }
      unsigned int index = triggerNames->triggerIndex(filter);
      if (index == triggerNames->size()) return false; //parsing failed
      filters.push_back(std::pair<unsigned int,bool>(index,flag));
   }
   if (filters.empty()) return false;

   TEventList* list = file.lists()[iSelector];
   list->Clear();

   // loop over events
   unsigned int iEvent = 0;
   for (file.event()->toBegin(); !file.event()->atEnd(); ++(*file.event()))
   {
      hTriggerResults.getByLabel(*file.event(),"TriggerResults","","HLT");
      std::vector<std::pair<unsigned int,bool> >::const_iterator filter = filters.begin();
      bool passed = hTriggerResults->accept(filter->first) == filter->second;
      ++filter;
      for (; filter != filters.end(); ++filter)
      {
         if (junction_mode)
            passed &= hTriggerResults->accept(filter->first) == filter->second;
         else
            passed |= hTriggerResults->accept(filter->first) == filter->second;
      }
      if (passed)
         list->Enter(iEvent);
      ++iEvent;
   }
   file.event()->to(currentEvent);
   return true;
}

//______________________________________________________________________________
// methods for gui state

bool
CmsShowNavigator::isFirstEvent()
{
   return m_currentEntry == 0;
}

bool
CmsShowNavigator::isLastEvent()
{
   bool last = kFALSE;
   if ( m_filterEvents )
   {
      if (m_currentFile == m_lastSelectedFile &&
          m_currentEntry == m_currentFile->mainSelection().GetN()-1)
         last = kTRUE;
   }
   else
   {
      if (m_currentFile == m_lastFile &&
          m_currentEntry == m_currentFile->tree()->GetEntries()-1)
         last = kTRUE;
   }
   return last;
}

void
CmsShowNavigator::showEventFilter(const TGWindow* p)
{
   if (m_guiFilter == 0)
   {
      m_guiFilter = new FWGUIEventFilter(p);
      m_guiFilter->m_applyAction->activated.connect(sigc::mem_fun(this, &CmsShowNavigator::applyFiltersFromGUI));
   }
   
   m_guiFilter->show(&m_selectors, *m_currentFile->event(), m_globalOR);
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

   for(int i=0; i<numberOfFilters; ++i) {
      FWEventSelector* selector = new FWEventSelector();
      {
         const FWConfiguration* value =
            iFrom.valueForKey( Form("EventFilter%d_enabled",i) );
         assert(value);
         std::istringstream s(value->value());
         s>>selector->enabled;
      }
      {
         const FWConfiguration* value =
            iFrom.valueForKey( Form("EventFilter%d_selection",i) );
         assert(value);
         std::istringstream s(value->value());
         s>>selector->selection;
      }
      {
         const FWConfiguration* value =
            iFrom.valueForKey( Form("EventFilter%d_comment",i) );
         assert(value);
         std::istringstream s(value->value());
         s>>selector->title;
      }
      m_selectors.push_back(selector);
   }
}

void
CmsShowNavigator::addTo(FWConfiguration& iTo) const
{
   int numberOfFilters(0);
   for (std::vector<FWEventSelector*>::const_iterator sel = m_selectors.begin();
        sel != m_selectors.end(); ++sel) {
      if ( (*sel)->removed ) continue;
      iTo.addKeyValue(Form("EventFilter%d_enabled",numberOfFilters),
                      FWConfiguration(Form("%d",(*sel)->enabled)));
      iTo.addKeyValue(Form("EventFilter%d_selection",numberOfFilters),
                      FWConfiguration((*sel)->selection));
      iTo.addKeyValue(Form("EventFilter%d_comment",numberOfFilters),
                      FWConfiguration((*sel)->title));
      ++numberOfFilters;
   }
   iTo.addKeyValue("EventFilter_total",FWConfiguration(Form("%d",numberOfFilters)));
   iTo.addKeyValue("EventFilter_enabled",FWConfiguration(Form("%d",m_filterEvents)));
}

