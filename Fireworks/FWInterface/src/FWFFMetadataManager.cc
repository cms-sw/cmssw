#include "Fireworks/FWInterface/src/FWFFMetadataManager.h"
#include "Fireworks/FWInterface/src/FWFFMetadataUpdateRequest.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"

#include "TClass.h"
FWFFMetadataManager::FWFFMetadataManager():
   m_event(0)
{
}

bool
FWFFMetadataManager::hasModuleLabel(std::string& iModuleLabel)
{
   if (m_event) {
      std::vector<edm::Provenance const *> provenances;
      m_event->getAllProvenance(provenances);

      for (size_t pi = 0, pe = provenances.size(); pi != pe; ++pi)
      {
         edm::Provenance const *provenance = provenances[pi];
         if (provenance && (provenance->branchDescription().moduleLabel() == iModuleLabel))
            return true;
      }
   }
   return false;
}

bool
FWFFMetadataManager::doUpdate(FWJobMetadataUpdateRequest* request)
{
   // Clean up previous data.
   usableData().clear();

   assert(m_typeAndReps);
   FWFFMetadataUpdateRequest *fullRequest = dynamic_cast<FWFFMetadataUpdateRequest*>(request);
   if (!fullRequest)
      return false;
   const edm::Event &event = fullRequest->event();
   m_event = &event;

   typedef std::set<std::string> Purposes;
   Purposes purposes;
   std::vector<edm::Provenance const *> provenances;

   event.getAllProvenance(provenances);

   for (size_t pi = 0, pe = provenances.size(); pi != pe; ++pi)
   {
      edm::Provenance const *provenance = provenances[pi];
      if (!provenance)
         continue;
      Data d;
      const edm::BranchDescription &desc = provenance->branchDescription();

      if (!desc.present())
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
         purposes.insert(infos[ii].purpose());
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
         
         // This is pretty much the same thing that happens 
         if (!FWItemAccessorFactory::classAccessedAsCollection(theClass))
         {
            fwLog(fwlog::kDebug) << theClass->GetName() 
                       << " will not be displayed in table." << std::endl;
              continue;
         }
         d.type_ = desc.fullClassName();
         d.purpose_ = *itPurpose;
         d.moduleLabel_ = desc.moduleLabel();
         d.productInstanceLabel_ = desc.productInstanceName();
         d.processName_ = desc.processName();
         usableData().push_back(d);
         fwLog(fwlog::kDebug) << "Add collection will display " << d.type_ 
                              << " " << d.moduleLabel_ 
                              << " " << d.productInstanceLabel_
                              << " " << d.processName_ << std::endl;
      }
   } 
   return true;
}

