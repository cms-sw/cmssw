// -*- C++ -*-
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.cc,v 1.57 2009/11/20 17:21:17 amraktad Exp $
//
#define private public
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/src/CmsShowMain.h"
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
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWConfiguration.h"

//
// constructors and destructor
//
CmsShowNavigator::CmsShowNavigator(const CmsShowMain &main):
   m_currentEvent(0),

   m_filtersEnabled(false),
   m_filterModeOR(true),
  
   m_filtersNeedUpdate(),

   m_maxNumberOfFilesToChain(1),

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
   try 
   {
      // delete all previous files
      while ( m_files.size() > 0 )
      {
         FWFileEntry* file = m_files.front();
         m_files.pop_front();
         file->closeFile();
         delete file;
      }
      
      FWFileEntry* newFile = new FWFileEntry(fileName);
      m_files.push_back(newFile);
      if (!m_currentFile.m_isSet)
         setCurrentFile(m_files.begin());
   
      if (m_filtersEnabled)
      {
         for (std::list<FWEventSelector*>::iterator i = m_selectors.begin(); i != m_selectors.end(); ++i)
         {
            newFile->filters().push_back(new FWFileEntry::Filter(*i));
         }
         m_filtersNeedUpdate = true;
         updateFileFilters();
      }
      return true;
   }
   catch (std::exception& iException) {
      std::cerr <<"Navigator::openFile caught exception "<<iException.what()<<std::endl;
      return false;
   }
}

bool
CmsShowNavigator::appendFile(const std::string& fileName, bool checkMaxFileSize)
{
   try
   {
      FWFileEntry* newFile = new FWFileEntry(fileName);
      if ( newFile->file() == 0 )
      {
         delete newFile;
         return false; //bad file
      }
   
      unsigned int nFilesKeep = checkMaxFileSize ? (m_maxNumberOfFilesToChain + 1) : 1;
      // remove extra files
      while ( m_files.size() > 0 && m_files.size() >= nFilesKeep &&
              m_files.begin() != m_currentFile)
      {
         FWFileEntry* file = m_files.front();
         file->closeFile();
         delete file;
         m_files.pop_front();
      }
   
      if (m_files.size() >= m_maxNumberOfFilesToChain)
         printf("WARNING:: %d chained files more than maxNumberOfFilesToChain [%d]\n", (int)m_files.size(), m_maxNumberOfFilesToChain);

      m_files.push_back(newFile);
      if (!m_currentFile.m_isSet)
         setCurrentFile(m_files.begin());

      if (m_filtersEnabled)
      {
         for (std::list<FWEventSelector*>::iterator i = m_selectors.begin(); i != m_selectors.end(); ++i)
         {
            newFile->filters().push_back(new FWFileEntry::Filter(*i));
         }
         m_filtersNeedUpdate = true;
         updateFileFilters();
      }
   }
   catch(std::exception& iException)
   {
      std::cerr <<"Navigator::openFile caught exception "<<iException.what()<<std::endl;
      return false;
   }

   return true;
}

//______________________________________________________________________________
void
CmsShowNavigator::eventFilterEnableCallback(Bool_t x)
{
   if (m_filtersEnabled == x)
      return;
   
   m_filtersEnabled = x;
   if (m_filtersEnabled)
   {
      m_filtersNeedUpdate = true;
      updateFileFilters();
   }
}

//______________________________________________________________________________
void CmsShowNavigator::setCurrentFile(FileQueue_i fi)
{
   m_currentFile = fi;
   fileChanged_.emit((*m_currentFile)->file());
}

void
CmsShowNavigator::goTo(FileQueue_i fi, int event)
{
   if (fi != m_currentFile)
      setCurrentFile(fi);

   (*m_currentFile)->event()->to(event);
   m_currentEvent = event;

   newEvent_.emit(*((*m_currentFile)->event()));
}

void
CmsShowNavigator::goToRunEvent(Int_t run, Int_t event)
{
   fwlite::Event* fwEvent = 0;
   edm::FileIndex::const_iterator it;

   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      fwEvent = (*file)->event();
      fwEvent->fillFileIndex();
      it = fwEvent->fileIndex_.findEventPosition(run, 0, event, true);
      if (fwEvent->fileIndex_.end() != it)
         goTo(file, event);
   }
}

//______________________________________________________________________________

void
CmsShowNavigator::firstEvent()
{
   FileQueue_i x = m_files.begin();
   if (m_filtersEnabled)
   {
       goTo(x,  (*x)->firstSelectedEvent());
   }
   else
   {
      goTo(x, 0);
   }
}

void
CmsShowNavigator::lastEvent()
{ 
   FileQueue_i x = m_files.end(); --x;
   if (m_filtersEnabled)
   {
      goTo(x, (*x)->lastSelectedEvent());
   }
   else
   {
      goTo(x, (*x)->lastEvent());
   }
}

//______________________________________________________________________________

bool
CmsShowNavigator::nextSelectedEvent()
{
   int nextEv = (*m_currentFile)->nextSelectedEvent(m_currentEvent);
   if (nextEv > -1)
   {
      goTo(m_currentFile, nextEv);
      return true;
   }
   else
   {
      FileQueue_i i = m_currentFile; ++i;
      while (i != m_files.end())
      {
         if ((*i)->hasSelectedEvents())
         {
            goTo(i, (*i)->firstSelectedEvent());
            return true;
         }
         ++i;
      }
   }
   
   return false;
}

//______________________________________________________________________________

void
CmsShowNavigator::nextEvent()
{
   if (m_filtersEnabled)
   {
      nextSelectedEvent();
      return;
   }
   else
   {
      if (m_currentEvent < (*m_currentFile)->lastEvent())
      {
         goTo(m_currentFile, m_currentEvent + 1);
      }
      else
      {
         FileQueue_i x = m_currentFile ; ++x;
         if (x != m_files.end())
         {  
            goTo(x, 0);
         }
      }
   }
}

//______________________________________________________________________________

bool
CmsShowNavigator::previousSelectedEvent()
{
   int prevEv = (*m_currentFile)->previousSelectedEvent(m_currentEvent);
   if (prevEv > -1)
   {
      goTo(m_currentFile, prevEv);
      return true;
   }
   else
   {
      FileQueue_i i(m_currentFile); i.previous(m_files);
      while (i != m_files.end())
      {
         if ((*i)->hasSelectedEvents())
         {
            goTo(i, (*i)->lastSelectedEvent());
            return true;
         }
         i.previous(m_files);
      }
   }
   return false;
}

//______________________________________________________________________________

void
CmsShowNavigator::previousEvent()
{
   if (m_filtersEnabled)
   {
      previousSelectedEvent();
   }
   else
   {
      if (m_currentEvent > 0)
      {
         goTo(m_currentFile, m_currentEvent - 1);
      }
      else
      {
         // last event in previous file
         FileQueue_i x = m_currentFile ; --x;
         if (x != m_files.begin())
         {  
            goTo(x, (*x)->lastEvent());
         }
      }
   }
}

//______________________________________________________________________________

void
CmsShowNavigator::updateFiltersEnabled(bool x)
{
   if (x != m_filtersEnabled)
   {
      m_filtersEnabled = x;
      if (m_filtersEnabled)
      {
         updateFileFilters();
      }
   }
}

void
CmsShowNavigator::updateFileFilters()
{
   // run filters on files
   std::list<FWFileEntry::Filter>::iterator it;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      if (m_filtersNeedUpdate) (*file)->filtersNeedUpdate();
      (*file)->updateFilters(m_main.m_eiManager.get(), m_filterModeOR);
   }
   m_filtersNeedUpdate = false;

   // go to nearest file
   if (!(*m_currentFile)->isEventSelected(m_currentEvent))
   { 
      if (!nextSelectedEvent())
         nextSelectedEvent();
   }

   int nSelected = getNSelectedEvents();
   if (nSelected)
   {     
      postFiltering_.emit();
   }
   else
   { 
     noEventSelected_.emit();
   }
}

//=======================================================================
void
CmsShowNavigator::removeFilter(std::list<FWEventSelector*>::iterator si)
{
   // printf("remove filter %s \n", (*si)->m_expression.c_str());

   std::list<FWFileEntry::Filter*>::iterator it;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      for (it = (*file)->filters().begin(); it != (*file)->filters().end(); ++it)
      {
         if ((*it)->m_selector == *si)
         {
            FWFileEntry::Filter* f = *it;
            delete f;
            (*file)->filters().erase(it);
            break;
         }
      }
   } 

   delete *si;
   m_selectors.erase(si);  
   m_filtersNeedUpdate = true;
}

void
CmsShowNavigator::addFilter(FWEventSelector* ref)
{
   //  printf("add filter %s\n", ref->m_expression.c_str());

   FWEventSelector* selector = new FWEventSelector(ref);
   m_selectors.push_back(selector);

   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      (*file)->filters ().push_back(new FWFileEntry::Filter(selector));
   }
   m_filtersNeedUpdate = true;
}

void
CmsShowNavigator::changeFilter(FWEventSelector* selector)
{
  // printf("change filter %s\n", selector->m_expression.c_str());

   std::list<FWFileEntry::Filter*>::iterator it;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      for (it = (*file)->filters().begin(); it != (*file)->filters().end(); ++it)
      {
         if ((*it)->m_selector == selector)
         {
            (*it)->m_needsUpdate = true;
            (*it)->m_selector->m_expression  = selector->m_expression;
            break;
         }

      } 
   }
   m_filtersNeedUpdate = true;
}

void
CmsShowNavigator::applyFiltersFromGUI()
{
   // compare changes and then call updateFileFilters

   std::list<FWEventSelector*>::iterator    si = m_selectors.begin();
   std::list<FWGUIEventSelector*>::iterator gi = m_guiFilter->guiSelectors().begin();

   m_filtersNeedUpdate = false;

   if (m_filterModeOR != m_guiFilter->isLogicalOR()) {  
      m_filterModeOR = m_guiFilter->isLogicalOR();
      m_filtersNeedUpdate = true;
   }
      
   while (si != m_selectors.end() || gi != m_guiFilter->guiSelectors().end())
   {
      if (gi == m_guiFilter->guiSelectors().end() && si != m_selectors.end())
      {
         removeFilter(si++);
      }
      else if (si == m_selectors.end() && gi != m_guiFilter->guiSelectors().end() )
      {
         addFilter((*gi)->guiSelector());
         (*gi)->setOrigSelector(m_selectors.back());
         ++gi;
      }
      else
      {
         if (*si == (*gi)->origSelector())
         {
            FWEventSelector* g  = (*gi)->guiSelector();
            FWEventSelector* o = *si;
            if (o->m_enabled != g->m_enabled)
               m_filtersNeedUpdate = true;
            if (o->m_expression != g->m_expression || o->m_enabled != g->m_enabled) {
               *o = *g;
               changeFilter(*si);
            }
            else
            {
               o->m_description = g->m_description;
            }
            ++si; ++gi;
         }
         else if ((*gi)->origSelector() == 0)
         {
            addFilter((*gi)->guiSelector());
            (*gi)->setOrigSelector(m_selectors.back());
            ++gi;
         }
         else
         {
            removeFilter(si++);
         }
      }
   }

   if (m_filtersNeedUpdate)
      updateFileFilters();
}



//______________________________________________________________________________
// helpers for gui state

bool
CmsShowNavigator::isFirstEvent()
{
   int first =  (m_filtersEnabled) ? (*m_files.begin())->firstSelectedEvent() : 0;
   return  first == m_currentEvent;
}

bool
CmsShowNavigator::isLastEvent()
{
   if (m_filtersEnabled)
   {
      FileQueue_i lastSelectedFile;
      for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
      {
         if ((*file)->hasSelectedEvents())
            lastSelectedFile = file;
      }
      if (lastSelectedFile == m_currentFile)
         return (*m_currentFile)->lastSelectedEvent() == m_currentEvent;
   }
   else
   {
      FileQueue_i lastFile = m_files.end();
      --lastFile;
      if (m_currentFile == lastFile)
      {
         return (*m_currentFile)->lastEvent() == m_currentEvent;
      }
   }
   return false;
}

//______________________________________________________________________________

int 
CmsShowNavigator::getNSelectedEvents()
{ 
   int sum = 0;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      sum += (*file)->globalSelection()->GetN();
   }
   return sum;      
}

int
CmsShowNavigator::getNTotalEvents()
{  
   int sum = 0;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      sum += (*file)->tree()->GetEntries();
   }

   return sum;      
}

const char*
CmsShowNavigator::filterStatusMessage()
{
   if (m_filtersEnabled)
      return Form("%d events are selected from %d", getNSelectedEvents(), getNTotalEvents());
   else
      return "Filtering is OFF.";
}

bool
CmsShowNavigator::canEditFiltersExternally()
{
   bool haveActiveFilters = false;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      if ((*file)->hasActiveFilters())
      {
         haveActiveFilters = true;
         break;
      }
   }

   bool btnEnabled = haveActiveFilters;

   if (m_guiFilter && m_guiFilter->isOpen()) 
      btnEnabled = false;

   return btnEnabled;
}

void
CmsShowNavigator::editFiltersExternally()
{
   editFiltersExternally_.emit(m_filtersEnabled, canEditFiltersExternally());
}

void
CmsShowNavigator::showEventFilterGUI(const TGWindow* p)
{
   if (m_guiFilter == 0)
   {
      m_guiFilter = new FWGUIEventFilter(p);
      m_guiFilter->m_applyAction->activated.connect(sigc::mem_fun(this, &CmsShowNavigator::applyFiltersFromGUI));
      m_guiFilter->m_finishEditAction->activated.connect(sigc::mem_fun(this, &CmsShowNavigator::editFiltersExternally));
   }
   
   if (m_guiFilter->IsMapped())
   {
      m_guiFilter->CloseWindow();
   }
   else
   {
      m_guiFilter->show(&m_selectors, (*m_currentFile)->event(), m_filterModeOR);
   
      editFiltersExternally_.emit(m_filtersEnabled, canEditFiltersExternally());
   }
}

//______________________________________________________________________________

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
      s>>m_filtersEnabled;
   }

   for(int i=0; i<numberOfFilters; ++i) {
      FWEventSelector* selector = new FWEventSelector();
      {
         const FWConfiguration* value =
            iFrom.valueForKey( Form("EventFilter%d_enabled",i) );
         assert(value);
         std::istringstream s(value->value());
         s>>selector->m_enabled;
      }
      {
         const FWConfiguration* value =
            iFrom.valueForKey( Form("EventFilter%d_selection",i) );
         assert(value);
         std::istringstream s(value->value());
         s>>selector->m_expression;
      }
      {
         const FWConfiguration* value =
            iFrom.valueForKey( Form("EventFilter%d_comment",i) );
         assert(value);
         std::istringstream s(value->value());
         s>>selector->m_description;
      }
      m_selectors.push_back(selector);
   }
}

void
CmsShowNavigator::addTo(FWConfiguration& iTo) const
{
   int numberOfFilters(0);
   for (std::list<FWEventSelector*>::const_iterator sel = m_selectors.begin();
        sel != m_selectors.end(); ++sel) {
      iTo.addKeyValue(Form("EventFilter%d_enabled",numberOfFilters),
                      FWConfiguration(Form("%d",(*sel)->m_enabled)));
      iTo.addKeyValue(Form("EventFilter%d_selection",numberOfFilters),
                      FWConfiguration((*sel)->m_expression));
      iTo.addKeyValue(Form("EventFilter%d_comment",numberOfFilters),
                      FWConfiguration((*sel)->m_description));
      ++numberOfFilters;
   }
   iTo.addKeyValue("EventFilter_total",FWConfiguration(Form("%d",numberOfFilters)));
   iTo.addKeyValue("EventFilter_enabled",FWConfiguration(Form("%d",m_filtersEnabled)));
}

