// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverFactoryProducer
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
#include "FWCore/Framework/interface/ESProductResolverFactoryProducer.h"
#include "FWCore/Framework/interface/ESProductResolverFactoryBase.h"

#include "FWCore/Framework/interface/ESProductResolver.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace edm::eventsetup;
namespace edm {

  using Record2Factories = std::multimap<EventSetupRecordKey, FactoryInfo>;

  ESProductResolverFactoryProducer::ESProductResolverFactoryProducer() : record2Factories_() {}

  ESProductResolverFactoryProducer::~ESProductResolverFactoryProducer() noexcept(false) {}

  ESProductResolverProvider::KeyedResolversVector ESProductResolverFactoryProducer::registerProxies(const EventSetupRecordKey& iRecord,
                                                                                unsigned int iovIndex) {
    KeyedResolversVector keyedResolversVector;
    using Iterator = Record2Factories::iterator;
    std::pair<Iterator, Iterator> range = record2Factories_.equal_range(iRecord);
    for (Iterator it = range.first; it != range.second; ++it) {
      std::shared_ptr<ESProductResolver> resolver(it->second.factory_->makeResolver(iovIndex).release());
      if (nullptr != resolver.get()) {
        keyedResolversVector.emplace_back((*it).second.key_, resolver);
      }
    }
    return keyedResolversVector;
  }

  void ESProductResolverFactoryProducer::registerFactoryWithKey(const EventSetupRecordKey& iRecord,
                                                      std::unique_ptr<ESProductResolverFactoryBase> iFactory,
                                                      const std::string& iLabel) {
    if (nullptr == iFactory.get()) {
      assert(false && "Factor pointer was null");
      ::exit(1);
    }

    usingRecordWithKey(iRecord);

    std::shared_ptr<ESProductResolverFactoryBase> temp(iFactory.release());
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
