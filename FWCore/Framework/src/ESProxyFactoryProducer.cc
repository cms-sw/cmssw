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
#include <algorithm>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/ESProxyFactoryProducer.h"
#include "FWCore/Framework/interface/ProxyFactoryBase.h"

#include "FWCore/Framework/interface/DataProxy.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace edm::eventsetup;
namespace edm {

  using Record2Factories = std::multimap<EventSetupRecordKey, FactoryInfo>;

  ESProxyFactoryProducer::ESProxyFactoryProducer() : record2Factories_() {}

  ESProxyFactoryProducer::~ESProxyFactoryProducer() noexcept(false) {}

  DataProxyProvider::KeyedProxiesVector ESProxyFactoryProducer::registerProxies(const EventSetupRecordKey& iRecord,
                                                                                unsigned int iovIndex) {
    KeyedProxiesVector keyedProxiesVector;
    using Iterator = Record2Factories::iterator;
    std::pair<Iterator, Iterator> range = record2Factories_.equal_range(iRecord);
    for (Iterator it = range.first; it != range.second; ++it) {
      std::shared_ptr<DataProxy> proxy(it->second.factory_->makeProxy(iovIndex).release());
      if (nullptr != proxy.get()) {
        keyedProxiesVector.emplace_back((*it).second.key_, proxy);
      }
    }
    return keyedProxiesVector;
  }

  void ESProxyFactoryProducer::registerFactoryWithKey(const EventSetupRecordKey& iRecord,
                                                      std::unique_ptr<ProxyFactoryBase> iFactory,
                                                      const std::string& iLabel) {
    if (nullptr == iFactory.get()) {
      assert(false && "Factor pointer was null");
      ::exit(1);
    }

    usingRecordWithKey(iRecord);

    std::shared_ptr<ProxyFactoryBase> temp(iFactory.release());
    FactoryInfo info(temp->makeKey(iLabel), temp);

    //has this already been registered?
    std::pair<Record2Factories::const_iterator, Record2Factories::const_iterator> range =
        record2Factories_.equal_range(iRecord);
    if (range.second !=
        std::find_if(range.first, range.second, [&info](const auto& r2f) { return r2f.second.key_ == info.key_; })) {
      throw cms::Exception("IdenticalProducts")
          << "Producer has been asked to produce " << info.key_.type().name() << " \"" << info.key_.name().value()
          << "\" multiple times.\n Please modify the code.";
    }

    record2Factories_.insert(Record2Factories::value_type(iRecord, std::move(info)));
  }

}  // namespace edm
