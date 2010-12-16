// -*- C++ -*-
//
// Package:     Services
// Class  :     PrintEventSetupDataRetrieval
// 
// Implementation:
//     A service which prints which data from the EventSetup have been retrieved since the last time it checked
//
// Original Author:  Chris Jones
//         Created:  Thu Jul  9 14:30:13 CDT 2009
//

// system include files
#include <iostream>

// user include files
#include "FWCore/Services/src/PrintEventSetupDataRetrieval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
   PrintEventSetupDataRetrieval::PrintEventSetupDataRetrieval(const ParameterSet& iPS,
                                                              ActivityRegistry&iRegistry):
   m_printProviders(iPS.getUntrackedParameter<bool>("printProviders"))
   {
      if(iPS.getUntrackedParameter<bool>("checkAfterBeginRun")) {
         iRegistry.watchPostBeginRun(this, &PrintEventSetupDataRetrieval::postBeginRun);
      }
      if(iPS.getUntrackedParameter<bool>("checkAfterBeginLumi")) {
         iRegistry.watchPostBeginLumi(this, &PrintEventSetupDataRetrieval::postBeginLumi);
      }
      if(iPS.getUntrackedParameter<bool>("checkAfterEvent")) {
         iRegistry.watchPostProcessEvent(this, &PrintEventSetupDataRetrieval::postProcessEvent);
      }
   }

// PrintEventSetupDataRetrieval::PrintEventSetupDataRetrieval(const PrintEventSetupDataRetrieval& rhs)
// {
//    // do actual copying here;
// }

   //PrintEventSetupDataRetrieval::~PrintEventSetupDataRetrieval()
   //{
   //}

   void PrintEventSetupDataRetrieval::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<bool>("printProviders",false)->setComment(
      "If 'true' also print which ES module provides the data");
      desc.addUntracked<bool>("checkAfterBeginRun",false)->setComment(
       "If 'true' check for retrieved data after each begin run is processed");
      desc.addUntracked<bool>("checkAfterBeginLumi",false)->setComment(
       "If 'true' check for retrieved data after each begin lumi is processed");
      desc.addUntracked<bool>("checkAfterEvent",true)->setComment(
       "If 'true' check for retrieved data after an event is processed");
      descriptions.add("PrintEventSetupDataRetrieval", desc);
      descriptions.setComment("This service reports when EventSetup data is retrieved by a module in the job.");
   }

//
// assignment operators
//
// const PrintEventSetupDataRetrieval& PrintEventSetupDataRetrieval::operator=(const PrintEventSetupDataRetrieval& rhs)
// {
//   //An exception safe implementation is
//   PrintEventSetupDataRetrieval temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
   void 
   PrintEventSetupDataRetrieval::postProcessEvent(Event const&, EventSetup const& iES) {
      check(iES);
   }

   void 
   PrintEventSetupDataRetrieval::postBeginRun(Run const&, EventSetup const& iES) {
      check(iES);
   }

   void 
   PrintEventSetupDataRetrieval::postBeginLumi(LuminosityBlock const&, EventSetup const& iES) {
      check(iES);
   }
   
   void PrintEventSetupDataRetrieval::check(EventSetup const& iES) {
      //std::cout <<"postProcessEvent"<<std::endl;
      m_recordKeys.clear();
      iES.fillAvailableRecordKeys(m_recordKeys);
      
      for(std::vector<eventsetup::EventSetupRecordKey>::const_iterator it = m_recordKeys.begin(), itEnd = m_recordKeys.end();
          it != itEnd;
          ++it) {
         //std::cout <<"  "<<it->name()<<std::endl;
         const eventsetup::EventSetupRecord* r = iES.find(*it);
         
         RetrievedDataMap::iterator itRetrievedData =  m_retrievedDataMap.find(*it);
         if(itRetrievedData == m_retrievedDataMap.end()) {
            itRetrievedData = m_retrievedDataMap.insert(std::make_pair(*it,std::pair<unsigned long long, std::map<eventsetup::DataKey,bool> >())).first;
            itRetrievedData->second.first = r->cacheIdentifier();
            std::vector<eventsetup::DataKey> keys;
            r->fillRegisteredDataKeys(keys);
            for(std::vector<eventsetup::DataKey>::const_iterator itData = keys.begin(), itDataEnd = keys.end();
                itData != itDataEnd;
                ++itData) {
               itRetrievedData->second.second.insert(std::make_pair(*itData,false));
            }
         }
         RetrievedDataMap::value_type& retrievedData = *itRetrievedData;
         if(itRetrievedData->second.first != r->cacheIdentifier()) {
            itRetrievedData->second.first = r->cacheIdentifier();
            for(std::map<eventsetup::DataKey,bool>::iterator itDatum = retrievedData.second.second.begin(), itDatumEnd = retrievedData.second.second.end();
                itDatum != itDatumEnd;
                ++itDatum) {
               itDatum->second = false;
            }
         }
         
         for(std::map<eventsetup::DataKey,bool>::iterator itDatum = retrievedData.second.second.begin(), itDatumEnd = retrievedData.second.second.end();
             itDatum != itDatumEnd;
             ++itDatum) {
            bool wasGotten = r->wasGotten(itDatum->first);
            //std::cout <<"     "<<itDatum->first.type().name()<<" "<<wasGotten<<std::endl;
            if(wasGotten != itDatum->second) {
               itDatum->second = wasGotten;
               if(m_printProviders) {
                  const edm::eventsetup::ComponentDescription* d = r->providerDescription(itDatum->first);
                  assert(0!=d);
                  edm::LogSystem("PrintEventSetupDataRetrieval")<<"Retrieved> Record:"<<it->name()<<" data:"<<itDatum->first.type().name()<<" '"<<itDatum->first.name().value()
                  <<"' provider:"<<d->type_<<" '"<<d->label_<<"'";
               } else {
                  edm::LogSystem("PrintEventSetupDataRetrieval")<<"Retrieved> Record:"<<it->name()<<" data:"<<itDatum->first.type().name()<<" '"<<itDatum->first.name().value()<<"'";
               }
               //std::cout <<"CHANGED"<<std::endl;
            }
         }
      }
   }

//
// const member functions
//

//
// static member functions
//
}
