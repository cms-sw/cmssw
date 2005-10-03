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
// $Id: ESProxyFactoryProducer.cc,v 1.6 2005/09/29 21:06:47 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESProxyFactoryProducer.h"
#include "FWCore/Framework/interface/ProxyFactoryBase.h"

#include "FWCore/Framework/interface/DataProxy.h"

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
ESProxyFactoryProducer::ESProxyFactoryProducer()
{
}

// ESProxyFactoryProducer::ESProxyFactoryProducer(const ESProxyFactoryProducer& rhs)
// {
//    // do actual copying here;
// }

ESProxyFactoryProducer::~ESProxyFactoryProducer()
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
      
      boost::shared_ptr<DataProxy> proxy(it->second.factory_->makeProxy().release());
      if(0 != proxy.get()) {
         iProxies.push_back(KeyedProxies::value_type((*it).second.key_,
                                         proxy));
      }
   }
}

void
ESProxyFactoryProducer::registerFactoryWithKey(const EventSetupRecordKey& iRecord ,
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
