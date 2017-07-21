// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProxyFactoryProducer
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Apr  7 21:36:15 CDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESProxyFactoryProducer.h"
#include "FWCore/Framework/interface/ProxyFactoryBase.h"

#include "FWCore/Framework/interface/DataProxy.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <cassert>
//
// constants, enums and typedefs
//
using namespace edm::eventsetup;
namespace edm {

      typedef std::multimap< EventSetupRecordKey, FactoryInfo > Record2Factories;

//
// static data member definitions
//

//
// constructors and destructor
//
ESProxyFactoryProducer::ESProxyFactoryProducer() : record2Factories_()
{
}

// ESProxyFactoryProducer::ESProxyFactoryProducer(const ESProxyFactoryProducer& rhs)
// {
//    // do actual copying here;
// }

ESProxyFactoryProducer::~ESProxyFactoryProducer() noexcept(false)
{
}

//
// assignment operators
//
// const ESProxyFactoryProducer& ESProxyFactoryProducer::operator=(const ESProxyFactoryProducer& rhs)
// {
//   //An exception safe implementation is
//   ESProxyFactoryProducer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
ESProxyFactoryProducer::registerProxies(const EventSetupRecordKey& iRecord,
                                       KeyedProxies& iProxies)
{
   typedef Record2Factories::iterator Iterator;
   std::pair< Iterator, Iterator > range = record2Factories_.equal_range(iRecord);
   for(Iterator it = range.first; it != range.second; ++it) {

      std::shared_ptr<DataProxy> proxy(it->second.factory_->makeProxy().release());
      if(nullptr != proxy.get()) {
         iProxies.push_back(KeyedProxies::value_type((*it).second.key_,
                                         proxy));
      }
   }
}

void
ESProxyFactoryProducer::registerFactoryWithKey(const EventSetupRecordKey& iRecord ,
                                             std::unique_ptr<ProxyFactoryBase> iFactory,
                                             const std::string& iLabel )
{
   if(nullptr == iFactory.get()) {
      assert(false && "Factor pointer was null");
      ::exit(1);
   }

   usingRecordWithKey(iRecord);

   std::shared_ptr<ProxyFactoryBase> temp(iFactory.release());
   FactoryInfo info(temp->makeKey(iLabel),
                    temp);

   //has this already been registered?
   std::pair<Record2Factories::const_iterator,Record2Factories::const_iterator> range =
      record2Factories_.equal_range(iRecord);
   if(range.second != std::find_if(range.first,range.second,
                                   [&info](const auto& r2f) {
                                     return r2f.second.key_ == info.key_;
                                   }
                                   ) ) {
      throw cms::Exception("IdenticalProducts")<<"Producer has been asked to produce "<<info.key_.type().name()
      <<" \""<<info.key_.name().value()<<"\" multiple times.\n Please modify the code.";
   }

   record2Factories_.insert(Record2Factories::value_type(iRecord,
                                                         std::move(info)));
}

void
ESProxyFactoryProducer::newInterval(const EventSetupRecordKey& iRecordType,
                                   const ValidityInterval& /*iInterval*/)
{
   invalidateProxies(iRecordType);
}

//
// const member functions
//

//
// static member functions
//
}
