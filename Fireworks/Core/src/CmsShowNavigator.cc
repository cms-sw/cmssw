// -*- C++ -*-
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.cc,v 1.73 2009/12/02 17:06:36 amraktad Exp $
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
#include "TError.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TBranch.h"

// user include files
#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/FWTEventList.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWConfiguration.h"

//
// constructors and destructor
//
CmsShowNavigator::CmsShowNavigator(const CmsShowMain &main):
   m_currentEvent(0),

   m_filterState(kOff),
   m_filterMode(kOr),

   m_filtersNeedUpdate(),
   m_newFileOnNextEvent(false),

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
      setCurrentFile(m_files.begin());

      // set filters
      for (std::list<FWEventSelector*>::iterator i = m_selectors.begin(); i != m_selectors.end(); ++i)
         newFile->filters().push_back(new FWFileEntry::Filter(*i));

      if (m_filterState != kOff)
         updateFileFilters();

      return true;
   }
   catch (std::exception& iException) {
      std::cerr <<"Navigator::openFile caught exception "<<iException.what()<<std::endl;
      return false;
   }
}

bool
CmsShowNavigator::appendFile(const std::string& fileName, bool checkFileQueueSize, bool live)
{
   try
   {
      FWFileEntry* newFile = new FWFileEntry(fileName);
      if ( newFile->file() == 0 )
      {
         delete newFile;
         return false; //bad file
      }
      
      if (checkFileQueueSize)
      {
         int toErase = m_files.size() - (m_maxNumberOfFilesToChain + 1);
         while (toErase > 0)
         {
            FileQueue_i si = m_files.begin();
            if (m_currentFile == si)
               si++;
            FWFileEntry* file = *si;
            file->closeFile();
            delete file;
            
            m_files.erase(si);
            --toErase;
         }

         if (m_files.size() >= m_maxNumberOfFilesToChain)
            printf("WARNING:: %d chained files more than maxNumberOfFilesToChain [%d]\n", (int)m_files.size(), m_maxNumberOfFilesToChain);
      }
      
      m_files.push_back(newFile);

      // Needed for proper handling of first registered file when -port option is in effect.
      if (!m_currentFile.m_isSet)
         setCurrentFile(m_files.begin());

      // set filters
      for (std::list<FWEventSelector*>::iterator i = m_selectors.begin(); i != m_selectors.end(); ++i)
         newFile->filters().push_back(new FWFileEntry::Filter(*i));

      if (m_filterState != kOff)     
         updateFileFilters();

   }   
   catch(std::exception& iException)
   {
      std::cerr <<"Navigator::openFile caught exception "<<iException.what()<<std::endl;
      return false;
   }

   return true;
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
         goTo(file, (*file)->getTreeEntryFromEventId(event));
   }
}

//______________________________________________________________________________

void
CmsShowNavigator::firstEvent()
{
   FileQueue_i x = m_files.begin();
   if (m_filterState == kOn)
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
   if (m_filterState == kOn)
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
   if (m_newFileOnNextEvent)
   {
      FileQueue_i last = m_files.end(); --last;
      if (m_filterState == kOn)
         goTo(last, (*last)->firstSelectedEvent());
      else
         goTo(last, 0);

      m_newFileOnNextEvent = false;
      return;
   }
   
   if (m_filterState == kOn)
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
   if (m_filterState == kOn)
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
CmsShowNavigator::toggleFilterEnable()
{
   // callback

   if (m_filterState == kOff)
   {
      m_filterState = kOn;
      if (m_guiFilter)
         m_guiFilter->m_filterDisableAction->enable();

      updateFileFilters();
   }
   else
   {
      m_filterState = kOff;
      if (m_guiFilter)
         m_guiFilter->m_filterDisableAction->disable();
   }

   filterStateChanged_.emit(m_filterState);
}

void
CmsShowNavigator::withdrawFilter()
{
   m_filterState = kWithdrawn;
   filterStateChanged_.emit(m_filterState);
}

void
CmsShowNavigator::resumeFilter()
{
   m_filterState = kOn;
   filterStateChanged_.emit(m_filterState);
}

void
CmsShowNavigator::updateFileFilters()
{
   // run filters on files
   std::list<FWFileEntry::Filter>::iterator it;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      if (m_filtersNeedUpdate) (*file)->filtersNeedUpdate();
      (*file)->updateFilters(m_main.m_eiManager.get(), m_filterMode == kOr);
   }
   updateSelectorsInfo();
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
      if (m_filterState == kWithdrawn)
         resumeFilter();
      postFiltering_.emit();
   }
   else
   {
      withdrawFilter();
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
   m_filtersNeedUpdate = false;

   // check if filters are set ON
    if (m_filterState == kOff)
   {
      m_filtersNeedUpdate = true;
      m_filterState = kOn;
      m_guiFilter->m_filterDisableAction->enable();
      filterStateChanged_.emit(m_filterState);
   }

   // compare changes and then call updateFileFilters
   std::list<FWEventSelector*>::iterator    si = m_selectors.begin();
   std::list<FWGUIEventSelector*>::iterator gi = m_guiFilter->guiSelectors().begin();

   if (m_filterMode != m_guiFilter->getFilterMode()) {
      m_filterMode = m_guiFilter->getFilterMode();
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
   if (m_filterState == kOn)
   {
      FileQueue_i firstSelectedFile;
      for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
      {
         if ((*file)->hasSelectedEvents())
         {
            firstSelectedFile = file;
            break;
         }
      }

      if (firstSelectedFile == m_currentFile)
         return (*m_currentFile)->firstSelectedEvent() == m_currentEvent;
   }
   else
   {
      if (m_currentFile == m_files.begin())
      {
         return m_currentEvent == 0;
      }
   }
   return false;
}

bool
CmsShowNavigator::isLastEvent()
{
   if (m_filterState == kOn)
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
void
CmsShowNavigator::updateSelectorsInfo()
{
 
   // reset
   std::list<FWEventSelector*>::const_iterator sel = m_selectors.begin();
   while ( sel != m_selectors.end())
   {
      (*sel)->m_selected = 0;
      ++sel;
   }

   // loop file filters
   std::list<FWFileEntry::Filter*>::iterator i;
   for (FileQueue_i file = m_files.begin(); file != m_files.end(); ++file)
   {
      std::list<FWFileEntry::Filter*>& filters = (*file)->filters();
      for (i = filters.begin(); i != filters.end(); ++i)
      {
         if ((*i)->m_eventList)
         {
            (*i)->m_selector->m_selected += (*i)->m_eventList->GetN();
         }
      }
   }
   if (m_guiFilter) 
   {
      std::list<FWGUIEventSelector*>::const_iterator gs = m_guiFilter->guiSelectors().begin();
      while ( gs !=  m_guiFilter->guiSelectors().end())
      {
         (*gs)->updateNEvents();
         ++gs;
      }
   }
}

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
   if (m_filterState == kOn)
      return Form("%d events are selected from %d.", getNSelectedEvents(), getNTotalEvents());
   else if (m_filterState == kOff)
      return "Filtering is OFF.";
   else
      return "Filtering is disabled.";
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
   editFiltersExternally_.emit(canEditFiltersExternally());
}

void
CmsShowNavigator::showEventFilterGUI(const TGWindow* p)
{
   if (m_guiFilter == 0)
   {
      m_guiFilter = new FWGUIEventFilter(p);
      m_guiFilter->m_applyAction->activated.connect(sigc::mem_fun(this, &CmsShowNavigator::applyFiltersFromGUI));
      m_guiFilter->m_filterDisableAction->activated.connect(sigc::mem_fun(this, &CmsShowNavigator::toggleFilterEnable));
      m_guiFilter->m_finishEditAction->activated.connect(sigc::mem_fun(this, &CmsShowNavigator::editFiltersExternally));
   }

   if (m_guiFilter->IsMapped())
   {
      m_guiFilter->CloseWindow();
   }
   else
   {
      m_guiFilter->show(&m_selectors, (*m_currentFile)->event(), m_filterMode);
      editFiltersExternally_.emit(canEditFiltersExternally());
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
      int x;
      s>> x;
      m_filterState = x ? kOn : kOff;
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
   iTo.addKeyValue("EventFilter_enabled",FWConfiguration(Form("%d", m_filterState == kOn ? 1 : 0)));
}

