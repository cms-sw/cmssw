#include <boost/regex.hpp>

#include "TFile.h"
#include "TTreeCache.h"
#include "TEveTreeTools.h"
#include "TError.h"
#include "TMath.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Common/interface/TriggerNames.h"

#define private public
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private

#include "Fireworks/Core/interface/FWFileEntry.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/fwPaths.h"

FWFileEntry::FWFileEntry(const std::string& name, bool checkVersion) :
   m_name(name), m_file(0), m_eventTree(0), m_event(0),
   m_needUpdate(true), m_globalEventList(0)
{
   openFile(checkVersion);
}

FWFileEntry::~FWFileEntry()
{
   for(std::list<Filter*>::iterator i = m_filterEntries.begin(); i != m_filterEntries.end(); ++i)
      delete (*i)->m_eventList;

   delete m_globalEventList;
}

void FWFileEntry::openFile(bool checkVersion)
{
   gErrorIgnoreLevel = 3000; // suppress warnings about missing dictionaries
   TFile *newFile = TFile::Open(m_name.c_str());
   if (newFile == 0 || newFile->IsZombie() || !newFile->Get("Events")) {
      //  std::cout << "Invalid file. Ignored." << std::endl;
      // return false;
      throw std::runtime_error("Invalid file. Ignored.");
   }
   gErrorIgnoreLevel = -1;
   m_file = newFile;


   // check CMSSW relese version for compatibility
   if (checkVersion) {
      typedef std::vector<edm::ProcessHistory> provList;
  
      TTree   *metaData = dynamic_cast<TTree*>(m_file->Get("MetaData"));
      TBranch *b = metaData->GetBranch("ProcessHistory");
      provList *x = 0;
      b->SetAddress(&x);
      b->GetEntry(0);
      
      const edm::ProcessConfiguration* dd = 0;
      int latestVersion =0;
      int currentVersionArr[] = {0, 0, 0};
      for (auto const& processHistory : *x)
      {
         for (auto const& processConfiguration : processHistory)
         {
            // std::cout << processConfiguration.releaseVersion() << "  " << processConfiguration.processName() << std::endl;
            TString dcv = processConfiguration.releaseVersion();
            fireworks::getDecomposedVersion(dcv, currentVersionArr);
            int nvv = currentVersionArr[0]*100 + currentVersionArr[1]*10 + currentVersionArr[2];
            if (nvv > latestVersion) {
               latestVersion = nvv;
               dd = &processConfiguration;
            }
         }
      }

      if (latestVersion) {
         fwLog(fwlog::kInfo) << "Checking process history. " << m_name.c_str() << " latest process \""  << dd->processName() << "\", version " << dd->releaseVersion() << std::endl;
    
         b->SetAddress(0);
         TString v = dd->releaseVersion();
         if (!fireworks::acceptDataFormatsVersion(v))
         {
            int* di = (fireworks::supportedDataFormatsVersion());
            TString msg = Form("incompatible data: Process version does not mactch major data formats version. File produced with %s. Data formats version \"CMSSW_%d_%d_%d\".\n", 
                               dd->releaseVersion().c_str(), di[0], di[1], di[2]);
            msg += "Use --no-version-check option if you still want to view the file.\n";
            throw std::runtime_error(msg.Data());
         }
      }
      else {
         TString msg = "No process history available\n";
         msg += "Use --no-version-check option if you still want to view the file.\n";
         throw std::runtime_error(msg.Data());
      }
   }

   // load event
   m_event = new fwlite::Event(m_file);

   if (m_event->size() == 0)
         throw std::runtime_error("fwlite::Event size == 0");


   m_eventTree = dynamic_cast<TTree*>(m_file->Get("Events"));

   if (m_eventTree == 0)
   { 
      throw std::runtime_error("Cannot find TTree 'Events' in the data file");
   }

   // This now set in DataHelper
   //TTreeCache::SetLearnEntries(2);
   //m_eventTree->SetCacheSize(10*1024*1024);
   //TTreeCache *tc = (TTreeCache*) m_file->GetCacheRead();
   //tc->AddBranch(m_event->auxBranch_,kTRUE);
   //tc->StartLearningPhase();
}

void FWFileEntry::closeFile()
{
   if (m_file) {
      printf("Reading %lld bytes in %d transactions.\n",
             m_file->GetBytesRead(), m_file->GetReadCalls());
      delete m_file->GetCacheRead(m_eventTree);

      m_file->Close();
      delete m_file;
   }
   if (m_event) delete m_event;
}

//______________________________________________________________________________

bool FWFileEntry::isEventSelected(int tree_entry)
{
   int idx = m_globalEventList->GetIndex(tree_entry);
   return idx >= 0;
}

bool FWFileEntry::hasSelectedEvents()
{
   return m_globalEventList->GetN() > 0;
}

int FWFileEntry::firstSelectedEvent()
{
   if (m_globalEventList->GetN() > 0)
   {
      return m_globalEventList->GetEntry(0);
   }
   else
   {
      return -1;
   }
}

int FWFileEntry::lastSelectedEvent()
{
   if (m_globalEventList->GetN() > 0)
      return m_globalEventList->GetEntry(m_globalEventList->GetN() - 1);
   else
      return -1;
}

int FWFileEntry::nextSelectedEvent(int tree_entry)
{
   // Find next selected event after the current one.
   // This returns the index in the selected event list.
   // If none exists -1 is returned.

   const Long64_t *list = m_globalEventList->GetList();
   Long64_t val = tree_entry;
   Long64_t idx = TMath::BinarySearch(m_globalEventList->GetN(), list, val);
   ++idx;
   if (idx >= m_globalEventList->GetN() || idx < 0)
      return -1;
   return list[idx];
}

int FWFileEntry::previousSelectedEvent(int tree_entry)
{
   // Find first selected event before current one.
   // This returns the index in the selected event list.
   // If none exists -1 is returned.

   const Long64_t *list = m_globalEventList->GetList();
   Long64_t val = tree_entry;
   Long64_t idx = TMath::BinarySearch(m_globalEventList->GetN(), list, val);
   if (list[idx] == val)
      --idx;
   if (idx >= 0)
      return list[idx];
   else
      return -1;
}

//______________________________________________________________________________
bool FWFileEntry::hasActiveFilters()
{
   for (std::list<Filter*>::iterator it = m_filterEntries.begin(); it != m_filterEntries.end(); ++it)
   {
      if ((*it)->m_selector->m_enabled)
         return true;
   }

   return false;
}

//______________________________________________________________________________
void FWFileEntry::updateFilters(const FWEventItemsManager* eiMng, bool globalOR)
{
   if (!m_needUpdate)
      return;
   
   if (m_globalEventList)
      m_globalEventList->Reset();
   else
      m_globalEventList = new FWTEventList;

   for (std::list<Filter*>::iterator it = m_filterEntries.begin(); it != m_filterEntries.end(); ++it)
   {
      if ((*it)->m_selector->m_enabled && (*it)->m_needsUpdate)
      {
         runFilter(*it, eiMng);
      }
      // Need to re-check if enabled after filtering as it can be set to false
      // in runFilter().
      if ((*it)->m_selector->m_enabled)
      {
         if ((*it)->hasSelectedEvents())
         {
            if (globalOR || m_globalEventList->GetN() == 0)
            {
               m_globalEventList->Add((*it)->m_eventList);
            }
            else
            {
               m_globalEventList->Intersect((*it)->m_eventList);
            }
         }
         else if (!globalOR)
         {
            m_globalEventList->Reset();
            break;
         }
      }
   }
   
   fwLog(fwlog::kDebug) << "FWFileEntry::updateFilters in [" << m_file->GetName() << "]  global selection [" << m_globalEventList->GetN() << "/" <<  m_eventTree->GetEntries() << "]" << std::endl;

   m_needUpdate = false;
}

//_____________________________________________________________________________
void FWFileEntry::runFilter(Filter* filter, const FWEventItemsManager* eiMng)
{
   if (!filter->m_selector->m_triggerProcess.empty())
   {
      filterEventsWithCustomParser(filter);
      return;
   }
    
   // parse selection for known Fireworks expressions
   std::string interpretedSelection = filter->m_selector->m_expression;

   for (FWEventItemsManager::const_iterator i = eiMng->begin(),
           end = eiMng->end(); i != end; ++i)
   {
      FWEventItem *item = *i;
      if (item == 0) 
         continue;
      // FIXME: hack to get full branch name filled
      if (item->m_event == 0) 
      {
         item->m_event = m_event;
         item->getPrimaryData();
         item->m_event = 0;
      }

      boost::regex re(std::string("\\$") + (*i)->name());

      if (boost::regex_search(interpretedSelection, re))
      {
         const edm::TypeWithDict elementType(const_cast<TClass*>(item->type()));
         const edm::TypeWithDict wrapperType = edm::TypeWithDict::byName(edm::wrappedClassName(elementType.name()));
         std::string fullBranchName = m_event->getBranchNameFor(wrapperType.typeInfo(),
                                                                item->moduleLabel().c_str(), 
                                                                item->productInstanceLabel().c_str(),
                                                                item->processName().c_str());

         interpretedSelection = boost::regex_replace(interpretedSelection, re,
                                                     fullBranchName + ".obj");

         // printf("selection after applying s/%s/%s/: %s\n",
         //     (std::string("\\$") + (*i)->name()).c_str(),
         //     ((*i)->m_fullBranchName + ".obj").c_str(),
         //     interpretedSelection.c_str());
      }
   }


   std::size_t found = interpretedSelection.find('$');
   if (found!=std::string::npos)
   {
       fwLog(fwlog::kError) << "FWFileEntry::RunFilter invalid expression " <<  interpretedSelection << std::endl;
       filter->m_needsUpdate = false;
       return;
   }

   m_file->cd();
   m_eventTree->SetEventList(0);
   
   // Since ROOT will leave any TBranches used in the filtering at the last event,
   // we need to be able to reset them to what fwlite::Event expects them to be
   // we do this by holding onto the old buffers and create temporary new ones.

   std::map<TBranch*, void*> prevAddrs;

   {
      TObjArray* branches = m_eventTree->GetListOfBranches();
      std::auto_ptr<TIterator> pIt( branches->MakeIterator());
      while (TObject* branchObj = pIt->Next())
      {
         TBranch* b = dynamic_cast<TBranch*> (branchObj);
         if (0!=b)
         {
            const char * name = b->GetName();
            unsigned int length = strlen(name);
            if (length > 1 && name[length-1] != '.')
            {
               // This is not a data branch so we should ignore it.
               continue;
            }
            if (0 != b->GetAddress())
            {
               if (prevAddrs.find(b) != prevAddrs.end())
               {
                  fwLog(fwlog::kWarning) << "FWFileEntry::runFilter branch is already in the map!\n";
               }
               prevAddrs.insert(std::make_pair(b, b->GetAddress()));

               // std::cout <<"Zeroing branch: "<< b->GetName() <<" "<< (void*) b->GetAddress() <<std::endl;
               b->SetAddress(0);
            }
         }
      }
   }

   if (filter->m_eventList)
      filter->m_eventList->Reset();
   else
      filter->m_eventList = new FWTEventList;

   TEveSelectorToEventList stoelist(filter->m_eventList, interpretedSelection.c_str());
   Long64_t result = m_eventTree->Process(&stoelist);

   if (result < 0)
      fwLog(fwlog::kWarning) << "FWFileEntry::runFilter in file [" << m_file->GetName() << "] filter [" << filter->m_selector->m_expression << "] is invalid." << std::endl;
   else      
      fwLog(fwlog::kDebug) << "FWFileEntry::runFilter is file [" << m_file->GetName() << "], filter [" << filter->m_selector->m_expression << "] has ["  << filter->m_eventList->GetN() << "] events selected" << std::endl;

   // Set back the old branch buffers.
   {
      for (auto i : prevAddrs)
      {
         // std::cout <<"Resetting branch: "<< i.first->GetName() <<" "<< i.second <<std::endl;
         i.first->SetAddress(i.second);
      }
   }

   filter->m_needsUpdate = false;
}

//______________________________________________________________________________

bool
FWFileEntry::filterEventsWithCustomParser(Filter* filterEntry)
{
   std::string selection(filterEntry->m_selector->m_expression);

   boost::regex re_spaces("\\s+");
   selection = boost::regex_replace(selection,re_spaces,"");
   if (selection.find("&&") != std::string::npos &&
       selection.find("||") != std::string::npos )
   {
      // Combination of && and || operators not supported.
      return false;
   }

   fwlite::Handle<edm::TriggerResults> hTriggerResults;
   edm::TriggerNames const* triggerNames(0);
   try
   {
      hTriggerResults.getByLabel(*m_event,"TriggerResults","", filterEntry->m_selector->m_triggerProcess.c_str());
      triggerNames = &(m_event->triggerNames(*hTriggerResults));
   }
   catch(...)
   {
      fwLog(fwlog::kWarning) << " failed to get trigger results with process name "<<  filterEntry->m_selector->m_triggerProcess << std::endl;
      return false;
   }
   
   // std::cout << "Number of trigger names: " << triggerNames->size() << std::endl;
   // for (unsigned int i=0; i<triggerNames->size(); ++i)
   //  std::cout << " " << triggerNames->triggerName(i);
   //std::cout << std::endl;
   
   bool junction_mode = true; // AND
   if (selection.find("||")!=std::string::npos)
      junction_mode = false; // OR

   boost::regex re("\\&\\&|\\|\\|");

   boost::sregex_token_iterator i(selection.begin(), selection.end(), re, -1);
   boost::sregex_token_iterator j;

   // filters and how they enter in the logical expression
   std::vector<std::pair<unsigned int,bool> > filters;

   while (i != j)
   {
      std::string filter = *i++;
      bool flag = true;
      if (filter[0] == '!')
      {
         flag = false;
         filter.erase(filter.begin());
      }
      unsigned int index = triggerNames->triggerIndex(filter);
      if (index == triggerNames->size()) 
      {
         // Trigger name not found.
         return false;
      }
      filters.push_back(std::make_pair(index, flag));
   }
   if (filters.empty())
      return false;

   if (filterEntry->m_eventList)
      filterEntry->m_eventList->Reset();
   else
       filterEntry->m_eventList = new FWTEventList();
   FWTEventList* list = filterEntry->m_eventList;

   // loop over events
   edm::EventID currentEvent = m_event->id();
   unsigned int iEvent = 0;

   for (m_event->toBegin(); !m_event->atEnd(); ++(*m_event))
   {
      hTriggerResults.getByLabel(*m_event,"TriggerResults","", filterEntry->m_selector->m_triggerProcess.c_str());
      std::vector<std::pair<unsigned int,bool> >::const_iterator filter = filters.begin();
      bool passed = hTriggerResults->accept(filter->first) == filter->second;
      while (++filter != filters.end())
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
   m_event->to(currentEvent);

   filterEntry->m_needsUpdate = false;
   
   fwLog(fwlog::kDebug) << "FWFile::filterEventsWithCustomParser file [" << m_file->GetName() << "], filter [" << filterEntry->m_selector->m_expression << "], selected [" << list->GetN() << "]"  << std::endl;
   
   return true;
}
