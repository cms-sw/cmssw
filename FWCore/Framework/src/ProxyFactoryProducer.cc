// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyFactoryProducer
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Apr  7 21:36:15 CDT 2005
// $Id: ProxyFactoryProducer.cc,v 1.5 2005/08/24 21:43:21 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProxyFactoryProducer.h"
#include "FWCore/Framework/interface/ProxyFactoryBase.h"

#include "FWCore/Framework/interface/DataProxy.h"

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {

      typedef std::multimap< EventSetupRecordKey, FactoryInfo > Record2Factories;

//
// static data member definitions
//

//
// constructors and destructor
//
ProxyFactoryProducer::ProxyFactoryProducer()
{
}

// ProxyFactoryProducer::ProxyFactoryProducer(const ProxyFactoryProducer& rhs)
// {
//    // do actual copying here;
// }

ProxyFactoryProducer::~ProxyFactoryProducer()
{
}

//
// assignment operators
//
// const ProxyFactoryProducer& ProxyFactoryProducer::operator=(const ProxyFactoryProducer& rhs)
// {
//   //An exception safe implementation is
//   ProxyFactoryProducer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
ProxyFactoryProducer::registerProxies(const EventSetupRecordKey& iRecord,
                                       KeyedProxies& iProxies)
{
   typedef Record2Factories::iterator Iterator;
   std::pair< Iterator, Iterator > range = record2Factories_.equal_range(iRecord);
   for(Iterator it = range.first; it != range.second; ++it) {
      
      boost::shared_ptr<DataProxy> proxy(it->second.factory_->makeProxy().release());
      if(0 != proxy.get()) {
         iProxies.push_back(KeyedProxies::value_type((*it).second.key_,
                                         proxy));
      }
   }
}

void
ProxyFactoryProducer::registerFactoryWithKey(const EventSetupRecordKey& iRecord ,
                                             std::auto_ptr<ProxyFactoryBase>& iFactory,
                                             const std::string& iLabel )
{
   if(0 == iFactory.get()) {
      assert(false && "Factor pointer was null");
      ::exit(1);
   }
   
   usingRecordWithKey(iRecord);
   
   boost::shared_ptr<ProxyFactoryBase> temp(iFactory.release());
   FactoryInfo info(temp->makeKey(iLabel),
                    temp);
   
   record2Factories_.insert(Record2Factories::value_type(iRecord,
                                                         info));
}

void 
ProxyFactoryProducer::newInterval(const EventSetupRecordKey& iRecordType,
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
}
