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
// $Id: CmsShowNavigator.cc,v 1.32 2009/08/12 13:29:30 amraktad Exp $
//

// hacks
#define private public
#include "DataFormats/FWLite/interface/Event.h"
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
#define private public
#include "CmsShowMain.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Provenance/interface/EventID.h"


//
// constructors and destructor
//
CmsShowNavigator::CmsShowNavigator(const CmsShowMain &main)
   : m_file(0),
     m_event(0),
     m_eventTree(0),
     m_eventList(0),
     m_currentEntry(0),
     m_lastEntry(-1),
     m_currentSelectedEntry(0),
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
      printf("Invalid file. Ignored.\n");
      return false;
   }
   CmsShowMain::resetFieldEstimate();
   if (m_file != 0) {
      delete m_eventList;
      delete m_eventTree;
      delete m_event;
      m_file->Close();
      delete m_file;
   }

   gErrorIgnoreLevel = -1;
   m_file = newFile;
   newFileLoaded_.emit(m_file);
   m_event = new fwlite::Event(m_file);
   m_eventTree = dynamic_cast<TTree*> (m_file->Get("Events"));
   assert(m_eventTree!=0);
   m_eventList = new TEventList("list","");
   filterEventsAndReset(m_selection); // first event is loaded at the end

   return true;
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
CmsShowNavigator::checkPositionInGoTo()
{
   if ( m_event->id() == m_firstID )
      atBeginning_.emit();
   if ( m_event->id() == m_lastID)
      atEnd_.emit();
}

void
CmsShowNavigator::nextEventChangeAlsoChangeFile(const std::string& fileName)
{
   if (m_file == 0)
      loadFile(fileName);
   else
      m_nextFile = fileName;
}


void
CmsShowNavigator::nextEvent()
{
   if( !m_nextFile.empty()) {
      bool loadedNewFile = loadFile(m_nextFile);
      m_nextFile.clear();
      if (loadedNewFile) return;
   }

   if (m_file)
   {
      if ( m_currentSelectedEntry == m_lastEntry ) {
         firstEvent();
         return;
      }

      if (m_currentSelectedEntry < m_lastEntry &&
          m_event->to(realEntry(m_currentSelectedEntry+1)) ) {
         ++m_currentSelectedEntry;
         newEvent_.emit(*m_event);
         if (m_currentSelectedEntry == m_lastEntry)
               atEnd_.emit();
      }
      else oldEvent_.emit(*m_event);
   }
}

void
CmsShowNavigator::previousEvent()
{
   if( !m_nextFile.empty()) {
      bool loadedNewFile = loadFile(m_nextFile);
      m_nextFile.clear();
      if (loadedNewFile) return;
   }

   if (m_file)
   {
      if ( m_currentSelectedEntry == 0 ) {
         lastEvent();
         return;
      }

      if (m_currentSelectedEntry > 0 &&
          m_event->to(realEntry(m_currentSelectedEntry-1)) ) {
         --m_currentSelectedEntry;
         newEvent_.emit(*m_event);
         if (m_currentSelectedEntry == 0)
            atBeginning_.emit();
      }
      else oldEvent_.emit(*m_event);
   }
}

void
CmsShowNavigator::firstEvent()
{
   m_currentSelectedEntry = 0;
   m_event->to(realEntry(m_currentSelectedEntry));
   newEvent_.emit(*m_event);
   atBeginning_.emit();
}

void
CmsShowNavigator::lastEvent()
{
   m_currentSelectedEntry = m_lastEntry;
   m_event->to(realEntry(m_currentSelectedEntry));
   newEvent_.emit(*m_event);
   atEnd_.emit();
}

void
CmsShowNavigator::goToRun(Int_t run)
{
   Int_t entry = realEntry(run, 0);
   if ( entry < 0 ) {
      oldEvent_.emit(*m_event);
      return;
   }
   Int_t index = entry;
   if (m_eventTree && m_eventTree->GetEventList() ) index = m_eventTree->GetEventList()->GetIndex(entry);
   if (m_event->to(entry)) {
      if ( index < 0 )
         std::cout << "WARNING: requested event is not among preselected events! " << std::endl;
      else
         m_currentSelectedEntry = index;
      newEvent_.emit(*m_event);
      checkPositionInGoTo();
   }
   else oldEvent_.emit(*m_event);
}

void
CmsShowNavigator::goToEvent(Int_t event)
{
   Int_t entry = realEntry(m_event->id().run(), event);
   if ( entry < 0 ) {
      oldEvent_.emit(*m_event);
      return;
   }
   Int_t index = entry;
   if (m_eventTree && m_eventTree->GetEventList() ) index = m_eventTree->GetEventList()->GetIndex(entry);
   if (m_event->to(entry)) {
      if ( index < 0 )
         std::cout << "WARNING: requested event is not among preselected events! " << std::endl;
      else
         m_currentSelectedEntry = index;
      newEvent_.emit(*m_event);
      checkPositionInGoTo();
   }
   else oldEvent_.emit(*m_event);
}

void
CmsShowNavigator::filterEventsAndReset(std::string selection)
{
   preFiltering_();
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
//        printf("selection after applying s/%s/%s/: %s\n",
//               (std::string("\\$") + (*i)->name()).c_str(),
//               (*i)->moduleLabel().c_str(),
//               new_sel.c_str());
      selection.swap(new_sel);
   }


//      std::string s = selection;
//      for (boost::sregex_iterator i = boost::sregex_iterator(s.begin(), s.end(), re),
//             end;
//        i != end;
//        ++i) {
//        printf("%s\n", i->str(0).c_str());
//      }
//      return;
   m_selection = selection;

   m_eventTree->SetEventList(0);
   if ( m_selection.length() != 0 ) {
      //since ROOT will leave any TBranches used in the filtering at the last event,
      // we need to be able to reset them to what fwlite::Event expects them to be
      // we do this by holding onto the old buffers and create temporary new ones

      TObjArray* branches = m_eventTree->GetListOfBranches();
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

      //        std::cout << "Selection requested: " << m_selection << std::endl;
      //NOTE: to be completely safe, we should disable the access to edm::Refs so that our
      // buffers do not get moved
      m_eventTree->Draw(">>list",m_selection.c_str());
      m_eventTree->SetEventList( m_eventList );

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


   m_lastEntry = m_event->size() -1;
   if ( m_eventTree->GetEventList() ){
      m_lastEntry = m_eventList->GetN()-1;
      if ( m_lastEntry < 0 ) {
         std::cout << "WARNING: No events passed selection: " << selection.c_str() << std::endl;
         m_eventTree->SetEventList(0);
         m_lastEntry = m_event->size() -1;
      }
   }
   postFiltering_();

   m_event->to(realEntry(0));
   m_firstID = m_event->id();
   m_event->to(realEntry(m_lastEntry));
   m_lastID = m_event->id();
   firstEvent();
}
