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
#include <string>
#include <exception>

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

#include "FWCore/Utilities/interface/ConvertException.h"
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
eventSetup_(nullptr),
cacheIdentifier_(1), //start with 1 since 0 means we haven't checked yet
transientAccessRequested_(false)
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

void
EventSetupRecord::getESProducers(std::vector<ComponentDescription const*>& esproducers) {
   esproducers.clear();
   esproducers.reserve(proxies_.size());
   for (auto const& iData : proxies_) {
      ComponentDescription const* componentDescription = iData.second->providerDescription();
      if (!componentDescription->isLooper_ && !componentDescription->isSource_) {
         esproducers.push_back(componentDescription);
      }
   }
}

void
EventSetupRecord::fillReferencedDataKeys(std::map<DataKey, ComponentDescription const*>& referencedDataKeys) {
   referencedDataKeys.clear();
   for (auto const& iData : proxies_) {
      referencedDataKeys[iData.first] = iData.second->providerDescription();
   }
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
EventSetupRecord::clearProxies() 
{
   proxies_.clear();
}

void 
EventSetupRecord::cacheReset() 
{
   transientAccessRequested_ = false;
   ++cacheIdentifier_;
}

bool
EventSetupRecord::transientReset()
{
   bool returnValue = transientAccessRequested_;
   transientAccessRequested_=false;
   return returnValue;
}
      
//
// const member functions
//
      
const void* 
EventSetupRecord::getFromProxy(DataKey const & iKey ,
                               const ComponentDescription*& iDesc,
                               bool iTransientAccessOnly) const
{
   if(iTransientAccessOnly) { this->transientAccessRequested(); }

   const DataProxy* proxy = this->find(iKey);
   
   const void* hold = 0;
   
   if(0!=proxy) {
      try {
         try {
            hold = proxy->get(*this, iKey,iTransientAccessOnly);
            iDesc = proxy->providerDescription(); 
         }
         catch (cms::Exception& e) { throw; }
         catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
         catch (std::exception& e) { convertException::stdToEDM(e); }
         catch(std::string& s) { convertException::stringToEDM(s); }
         catch(char const* c) { convertException::charPtrToEDM(c); }
         catch (...) { convertException::unknownToEDM(); }
      }
      catch(cms::Exception& e) {
         addTraceInfoToCmsException(e,iKey.name().value(),proxy->providerDescription(), iKey);
         //NOTE: the above function can't do the 'throw' since it causes the C++ class type
         // of the throw to be changed, a 'rethrow' does not have that problem
         throw;
      }
   }
   return hold;   
}
      
const DataProxy* 
EventSetupRecord::find(const DataKey& iKey) const 
{
   Proxies::const_iterator entry(proxies_.find(iKey)) ;
   if (entry != proxies_.end()) {
      return entry->second;
   }
   return 0;
}
      
bool 
EventSetupRecord::doGet(const DataKey& aKey, bool aGetTransiently) const {
   const DataProxy* proxy = find(aKey);
   if(0 != proxy) {
      try {
         try {
            proxy->doGet(*this, aKey, aGetTransiently);
         }
         catch (cms::Exception& e) { throw; }
         catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
         catch (std::exception& e) { convertException::stdToEDM(e); }
         catch(std::string& s) { convertException::stringToEDM(s); }
         catch(char const* c) { convertException::charPtrToEDM(c); }
         catch (...) { convertException::unknownToEDM(); }
      }
      catch( cms::Exception& e) {
         addTraceInfoToCmsException(e,aKey.name().value(),proxy->providerDescription(), aKey);
         //NOTE: the above function can't do the 'throw' since it causes the C++ class type
         // of the throw to be changed, a 'rethrow' does not have that problem
         throw;
      }
   }
   return 0 != proxy;
}

bool 
EventSetupRecord::wasGotten(const DataKey& aKey) const {
   const DataProxy* proxy = find(aKey);
   if(0 != proxy) {
      return proxy->cacheIsValid();
   }
   return false;
}

edm::eventsetup::ComponentDescription const* 
EventSetupRecord::providerDescription(const DataKey& aKey) const {
   const DataProxy* proxy = find(aKey);
   if(0 != proxy) {
      return proxy->providerDescription();
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
   std::ostringstream ost;
   ost << "Using EventSetup component "
       << iDescription->type_
       << "/'" << iDescription->label_
       << "' to make data "
       << iKey.type().name() << "/'"
       << iName
       << "' in record "
       << this->key().type().name();
   iException.addContext(ost.str());
}         

//
// static member functions
//
   }
}
