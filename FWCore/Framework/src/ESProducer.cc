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
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Utilities/interface/ESIndices.h"

namespace edm {

  ESProducer::ESProducer() : consumesInfos_{}, acquirer_{{{std::make_shared<SerialTaskQueue>()}}} {}

  ESProducer::~ESProducer() noexcept(false) {}

  void ESProducer::updateLookup(eventsetup::ESRecordsToProductResolverIndices const& iResolverToIndices) {
    if (sharedResourceNames_) {
      auto instance = SharedResourcesRegistry::instance();
      acquirer_ = instance->createAcquirer(*sharedResourceNames_);
      sharedResourceNames_.reset();
    }

    if (itemsToGetFromRecords_.size() == consumesInfos_.size()) {
      return;
    }

    itemsToGetFromRecords_.reserve(consumesInfos_.size());
    recordsUsedDuringGet_.reserve(consumesInfos_.size());

    for (auto& info : consumesInfos_) {
      auto& items = itemsToGetFromRecords_.emplace_back();
      items.reserve(info->size());
      auto& records = recordsUsedDuringGet_.emplace_back();
      records.reserve(info->size());
      for (auto& resolverInfo : *info) {
        //check for mayConsumes
        if (auto chooser = resolverInfo.chooser_.get()) {
          hasMayConsumes_ = true;
          auto tagGetter = iResolverToIndices.makeTagGetter(chooser->recordKey(), chooser->productType());
          if (not tagGetter.hasNothingToGet()) {
            records.push_back(iResolverToIndices.recordIndexFor(chooser->recordKey()));
          } else {
            //The record is not actually missing but the resolver is
            records.emplace_back(eventsetup::ESRecordsToProductResolverIndices::missingRecordIndex());
          }
          chooser->setTagGetter(std::move(tagGetter));
          // This value will get overwritten before being used
          items.push_back(ESResolverIndex::noResolverConfigured());
        } else {
          auto index = iResolverToIndices.indexInRecord(resolverInfo.recordKey_, resolverInfo.productKey_);
          if (index != ESResolverIndex::noResolverConfigured()) {
            if (not resolverInfo.moduleLabel_.empty()) {
              auto component = iResolverToIndices.component(resolverInfo.recordKey_, resolverInfo.productKey_);
              if (nullptr == component) {
                index = ESResolverIndex::moduleLabelDoesNotMatch();
              } else {
                if (component->label_.empty()) {
                  if (component->type_ != resolverInfo.moduleLabel_) {
                    index = ESResolverIndex::moduleLabelDoesNotMatch();
                  }
                } else if (component->label_ != resolverInfo.moduleLabel_) {
                  index = ESResolverIndex::moduleLabelDoesNotMatch();
                }
              }
            }
          }
          items.push_back(index);
          if (index != ESResolverIndex::noResolverConfigured() && index != ESResolverIndex::moduleLabelDoesNotMatch()) {
            records.push_back(iResolverToIndices.recordIndexFor(resolverInfo.recordKey_));
          } else {
            //The record is not actually missing but the resolver is
            records.emplace_back(eventsetup::ESRecordsToProductResolverIndices::missingRecordIndex());
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
