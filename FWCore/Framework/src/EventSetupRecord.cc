// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecord
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Mar 26 18:06:32 EST 2005
//

// system include files
#include <assert.h>

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
   namespace eventsetup {
//
// constants, enums and typedefs
//
      typedef std::map< DataKey , const DataProxy* > Proxies;
//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupRecord::EventSetupRecord() :
validity_(),
proxies_(),
eventSetup_(0),
cacheIdentifier_(1) //start with 1 since 0 means we haven't checked yet
{
}

// EventSetupRecord::EventSetupRecord(const EventSetupRecord& rhs)
// {
//    // do actual copying here;
// }

EventSetupRecord::~EventSetupRecord()
{
}

//
// assignment operators
//
// const EventSetupRecord& EventSetupRecord::operator=(const EventSetupRecord& rhs)
// {
//   //An exception safe implementation is
//   EventSetupRecord temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
EventSetupRecord::set(const ValidityInterval& iInterval) 
{
   validity_ = iInterval;
}

bool 
EventSetupRecord::add(const DataKey& iKey ,
                    const DataProxy* iProxy)
{
   //
   const DataProxy* proxy = find(iKey);
   if (0 != proxy) {
      //
      // we already know the field exist, so do not need to check against end()
      //
      
      // POLICY: If a Producer and a Source both claim to deliver the same data, the
      //  Producer 'trumps' the Source. If two modules of the same type claim to deliver the
      //  same data, this is an error unless the configuration specifically states which one
      //  is to be chosen.  A Looper trumps both a Producer and a Source.

      assert(proxy->providerDescription());
      assert(iProxy->providerDescription());
      if(iProxy->providerDescription()->isLooper_) {
         (*proxies_.find(iKey)).second = iProxy ;
	 return true;
      }
	 
      if(proxy->providerDescription()->isSource_ == iProxy->providerDescription()->isSource_) {
         //should lookup to see if there is a specified 'chosen' one and only if not, throw the exception
         throw cms::Exception("EventSetupConflict") <<"two EventSetup "<< 
         (proxy->providerDescription()->isSource_? "Sources":"Producers")
         <<" want to deliver type=\""<< iKey.type().name() <<"\" label=\""<<iKey.name().value()<<"\"\n"
         <<" from record "<<key().type().name() <<". The two providers are \n"
         <<"1) type=\""<<proxy->providerDescription()->type_<<"\" label=\""<<proxy->providerDescription()->label_<<"\"\n"
         <<"2) type=\""<<iProxy->providerDescription()->type_<<"\" label=\""<<iProxy->providerDescription()->label_<<"\"\n"
         <<"Please either\n   remove one of these "<<(proxy->providerDescription()->isSource_?"Sources":"Producers")
         <<"\n   or find a way of configuring one of them so it does not deliver this data"
         <<"\n   or use an es_prefer statement in the configuration to choose one.";
      } else if(proxy->providerDescription()->isSource_) {
         (*proxies_.find(iKey)).second = iProxy ;
      } else {
         return false;
      }
   }
   else {
      proxies_.insert(Proxies::value_type(iKey , iProxy)) ;
   }
   return true ;
}

void 
EventSetupRecord::cacheReset() 
{
  ++cacheIdentifier_;
}

//
// const member functions
//
const DataProxy* 
EventSetupRecord::find(const DataKey& iKey) const 
{
   Proxies::const_iterator entry(proxies_.find(iKey)) ;
   if (entry != proxies_.end()) {
      return entry->second;
   }
   return 0;
}

void 
EventSetupRecord::fillRegisteredDataKeys(std::vector<DataKey>& oToFill) const
{
  oToFill.clear();
  oToFill.reserve(proxies_.size());
  
  for(std::map< DataKey , const DataProxy* >::const_iterator it = proxies_.begin(), itEnd=proxies_.end();
      it != itEnd;
      ++it) {
    oToFill.push_back(it->first);
  }
  
}

void 
EventSetupRecord::validate(const ComponentDescription* iDesc, const ESInputTag& iTag) const
{
   if(iDesc && iTag.module().size()) {
      bool matched = false;
      if(iDesc->label_.empty()) {
         matched = iDesc->type_ == iTag.module();
      } else {
         matched = iDesc->label_ == iTag.module();
      }
      if(!matched) {
         throw cms::Exception("EventSetupWrongModule") <<"EventSetup data was retrieved using an ESInputTag with the values\n"
         <<"  moduleLabel = '"<<iTag.module()<<"'\n"
         <<"  dataLabel = '"<<iTag.data()<<"'\n"
         <<"but the data matching the C++ class type and dataLabel comes from module type="<<iDesc->type_<<" label='"<<iDesc->label_
         <<"'.\n Please either change the ESInputTag's 'module' label to be "<<( iDesc->label_.empty()? iDesc->type_:iDesc->label_)
         <<"\n or add the EventSetup module "<<iTag.module()<<" to the configuration.";
      }
   }
}

void 
EventSetupRecord::addTraceInfoToCmsException(cms::Exception& iException, const char* iName, const ComponentDescription* iDescription, const DataKey& iKey) const
{
   iException<<"cms::Exception going through EventSetup component "
   <<iDescription->type_
   <<"/\""<<iDescription->label_<<"\"\n"
   <<"  while making data "<< iKey.type().name()<<"/\""<<iName
   <<" in record \""<<this->key().type().name()<<"\"\n";
}         
      
void 
EventSetupRecord::changeStdExceptionToCmsException(const char* iExceptionWhatMessage, 
                                                   const char* iName, 
                                                   const ComponentDescription* iDescription, 
                                                   const DataKey& iKey) const
{
   cms::Exception changedException("StdException");
   changedException
   << "std::exception going through EventSetup component "
   <<iDescription->type_<<"/\""<<iDescription->label_<<"\"\n"
   <<"  while making data "<< iKey.type().name()<<"/\""<<iName<<" in record \""<<this->key().type().name()<<"\"\n"
   <<"  Previous information:\n  \"" << iExceptionWhatMessage<<"\"\n";
   throw changedException;
   
}         

//
// static member functions
//
   }
}
