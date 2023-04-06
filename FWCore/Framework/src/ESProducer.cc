// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProducer
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Apr 16 10:19:37 EDT 2005
//

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"

namespace edm {

  ESProducer::ESProducer() : consumesInfos_{}, acquirer_{{{std::make_shared<SerialTaskQueue>()}}} {}

  ESProducer::~ESProducer() noexcept(false) {}

  void ESProducer::updateLookup(eventsetup::ESRecordsToProxyIndices const& iProxyToIndices) {
    if (sharedResourceNames_) {
      auto instance = SharedResourcesRegistry::instance();
      acquirer_ = instance->createAcquirer(*sharedResourceNames_);
      sharedResourceNames_.reset();
    }

    itemsToGetFromRecords_.reserve(consumesInfos_.size());
    recordsUsedDuringGet_.reserve(consumesInfos_.size());

    if (itemsToGetFromRecords_.size() == consumesInfos_.size()) {
      return;
    }

    for (auto& info : consumesInfos_) {
      auto& items = itemsToGetFromRecords_.emplace_back();
      items.reserve(info->size());
      auto& records = recordsUsedDuringGet_.emplace_back();
      records.reserve(info->size());
      for (auto& proxyInfo : *info) {
        //check for mayConsumes
        if (auto chooser = proxyInfo.chooser_.get()) {
          hasMayConsumes_ = true;
          auto tagGetter = iProxyToIndices.makeTagGetter(chooser->recordKey(), chooser->productType());
          if (not tagGetter.hasNothingToGet()) {
            records.push_back(iProxyToIndices.recordIndexFor(chooser->recordKey()));
          } else {
            //The record is not actually missing but the proxy is
            records.emplace_back(eventsetup::ESRecordsToProxyIndices::missingRecordIndex());
          }
          chooser->setTagGetter(std::move(tagGetter));
          items.push_back(eventsetup::ESRecordsToProxyIndices::missingProxyIndex());
        } else {
          auto index = iProxyToIndices.indexInRecord(proxyInfo.recordKey_, proxyInfo.productKey_);
          if (index != eventsetup::ESRecordsToProxyIndices::missingProxyIndex()) {
            if (not proxyInfo.moduleLabel_.empty()) {
              auto component = iProxyToIndices.component(proxyInfo.recordKey_, proxyInfo.productKey_);
              if (nullptr == component) {
                index = eventsetup::ESRecordsToProxyIndices::missingProxyIndex();
              } else {
                if (component->label_.empty()) {
                  if (component->type_ != proxyInfo.moduleLabel_) {
                    index = eventsetup::ESRecordsToProxyIndices::missingProxyIndex();
                  }
                } else if (component->label_ != proxyInfo.moduleLabel_) {
                  index = eventsetup::ESRecordsToProxyIndices::missingProxyIndex();
                }
              }
            }
          }
          items.push_back(index);
          if (index != eventsetup::ESRecordsToProxyIndices::missingProxyIndex()) {
            records.push_back(iProxyToIndices.recordIndexFor(proxyInfo.recordKey_));
          } else {
            //The record is not actually missing but the proxy is
            records.emplace_back(eventsetup::ESRecordsToProxyIndices::missingRecordIndex());
          }
          assert(items.size() == records.size());
        }
      }
    }
  }

  void ESProducer::usesResources(std::vector<std::string> const& iResourceNames) {
    auto instance = SharedResourcesRegistry::instance();
    if (not sharedResourceNames_ and !iResourceNames.empty()) {
      sharedResourceNames_ = std::make_unique<std::vector<std::string>>(iResourceNames);
    }

    for (auto const& r : iResourceNames) {
      instance->registerSharedResource(r);
    }
    //Must defer setting acquirer_ until all resources have been registered
    // by all modules in the job
  }

}  // namespace edm
