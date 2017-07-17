#include "Fireworks/Core/interface/FWLiteJobMetadataManager.h"
#include "Fireworks/Core/interface/FWLiteJobMetadataUpdateRequest.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/FWLite/interface/Event.h"

#include "TFile.h"
#include "TTree.h"
#include <set>

FWLiteJobMetadataManager::FWLiteJobMetadataManager(void)
   : FWJobMetadataManager(),
     m_event(0)
{}

bool
FWLiteJobMetadataManager::hasModuleLabel(std::string& moduleLabel)
{
   if (m_event) {
      for ( auto bit = m_event->getBranchDescriptions().begin(); bit !=  m_event->getBranchDescriptions().end(); ++bit)
      {
         if (bit->moduleLabel() == moduleLabel) {
            return true;
         }
      }
   }
   return false;
}


/** This method inspects the currently opened TFile and for each branch 
    containing products for which we can either build a TCollectionProxy or 
    for which we have a specialized accessor, it registers it as a viewable 
    item.
 */
bool
FWLiteJobMetadataManager::doUpdate(FWJobMetadataUpdateRequest *request)
{
   FWLiteJobMetadataUpdateRequest *liteRequest 
      = dynamic_cast<FWLiteJobMetadataUpdateRequest *>(request);
   // There is no way we are going to get a non-FWLite updated request for
   // this class.
   assert(liteRequest);
   if (m_event == liteRequest->event_) 
      return false;

   m_event = liteRequest->event_;
   const TFile *file = liteRequest->file_;

   assert(file);
   
   usableData().clear();
   
   if (!m_event)
      return true;
   
   const std::vector<std::string>& history = m_event->getProcessHistory();
   
   // Turns out, in the online system we do sometimes gets files without any  
   // history, this really should be investigated
   if (history.empty())
      std::cout << "WARNING: the file '"
         << file->GetName() << "' contains no processing history"
            " and therefore should have no accessible data.\n";
   
   std::copy(history.rbegin(),history.rend(),
             std::back_inserter(processNamesInJob()));
   
   static const std::string s_blank;
   const std::vector<edm::BranchDescription>& descriptions =
      m_event->getBranchDescriptions();

   Data d;
   
   //I'm not going to modify TFile but I need to see what it is holding
   TTree* eventsTree = dynamic_cast<TTree*>(const_cast<TFile*>(file)->Get("Events"));
   assert(eventsTree);
   
   std::set<std::string> branchNamesInFile;
   TIter nextBranch(eventsTree->GetListOfBranches());
   while(TBranch* branch = static_cast<TBranch*>(nextBranch()))
      branchNamesInFile.insert(branch->GetName());
   
   typedef std::set<std::string> Purposes;
   Purposes purposes;
   std::string classType;
   
   for(size_t bi = 0, be = descriptions.size(); bi != be; ++bi) 
   {
      const edm::BranchDescription &desc = descriptions[bi];
      
      if (!desc.present() 
          || branchNamesInFile.end() == branchNamesInFile.find(desc.branchName()))
         continue;
      
      const std::vector<FWRepresentationInfo>& infos 
         = m_typeAndReps->representationsForType(desc.fullClassName());
      
      /*
      //std::cout <<"try to find match "<<itBranch->fullClassName()<<std::endl;
      //For each view we need to find the non-sub-part builder whose proximity is smallest and 
      // then register only that purpose
      //NOTE: for now, we will ignore the view and only look for the closest proximity
      unsigned int minProx = ~(0U);
      for (size_t ii = 0, ei = infos.size(); ii != ei; ++ii) {
      if (!infos[ii].representsSubPart() && minProx > infos[ii].proximity()) {
      minProx = infos[ii].proximity();
      }
      }
      */
      
      //the infos list can contain multiple items with the same purpose so we will just find
      // the unique ones
      purposes.clear();
      for (size_t ii = 0, ei = infos.size(); ii != ei; ++ii) {
         /* if(!infos[ii].representsSubPart() && minProx != infos[ii].proximity()) {
            continue;
            } */
         if (infos[ii].requiresFF() == false) {
             purposes.insert(infos[ii].purpose());
         }
      }
      
      if (purposes.empty())
         purposes.insert("Table");
      
      for (Purposes::const_iterator itPurpose = purposes.begin(),
              itEnd = purposes.end();
           itPurpose != itEnd;
           ++itPurpose) 
      {
         // Determine whether or not the class can be iterated
         // either by using a TVirtualCollectionProxy (of the class 
         // itself or on one of its members), or by using a 
         // FWItemAccessor plugin.
         TClass* theClass = TClass::GetClass(desc.fullClassName().c_str());
         
         if (!theClass)
            continue;
      
         if (!theClass->GetTypeInfo())
            continue;
         
         const static bool debug = false;
         // This is pretty much the same thing that happens 
         if (!FWItemAccessorFactory::classAccessedAsCollection(theClass) )
         {
            if (debug) {
               fwLog(fwlog::kDebug) << theClass->GetName() 
                                    << " will not be displayed in table." << std::endl;
            }
            continue;
         }
         d.type_ = desc.fullClassName();
         d.purpose_ = *itPurpose;
         d.moduleLabel_ = desc.moduleLabel();
         d.productInstanceLabel_ = desc.productInstanceName();
         d.processName_ = desc.processName();
         usableData().push_back(d);
         if (debug)
         {
            fwLog(fwlog::kDebug) << "Add collection will display " << d.type_ 
                                 << " " << d.moduleLabel_ 
                                 << " " << d.productInstanceLabel_
                                 << " " << d.processName_ << std::endl;
         }
      }
   }
   return true;
}
