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
#include <cassert>
#include <string>
#include <exception>

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
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
EventSetupRecord::EventSetupRecord()
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
//
// const member functions
//
      
bool
EventSetupRecord::doGet(const DataKey& aKey, bool aGetTransiently) const {
  return impl_->doGet(aKey,aGetTransiently);
}

bool 
EventSetupRecord::wasGotten(const DataKey& aKey) const {
  return impl_->wasGotten(aKey);
}

edm::eventsetup::ComponentDescription const* 
EventSetupRecord::providerDescription(const DataKey& aKey) const {
  return impl_->providerDescription(aKey);
}

void 
EventSetupRecord::validate(const ComponentDescription* iDesc, const ESInputTag& iTag) const
{
   if(iDesc && !iTag.module().empty()) {
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
