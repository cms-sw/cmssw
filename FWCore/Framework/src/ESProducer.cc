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
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESModuleConsumesInfo.h"
#include "FWCore/Utilities/interface/ESIndices.h"

#include <cassert>
#include <set>
#include <string_view>

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

  void ESProducer::esModulesWhoseProductsAreConsumed(std::vector<eventsetup::ComponentDescription const*>& esModules,
                                                     eventsetup::ESRecordsToProductResolverIndices const& iPI) const {
    std::set<unsigned int> alreadyFound;

    // updateLookup should have already been called if we are here
    // If it has been called, then this assert should not fail.
    assert(consumesInfos_.size() == itemsToGetFromRecords_.size());

    // Here transition identifies which call to setWhatProduced (each corresponding to a "produce" function)
    // Often there is only one.
    auto it = itemsToGetFromRecords_.begin();
    for (auto const& transition : consumesInfos_) {
      auto itResolver = it->begin();
      for (auto const& esConsumesInfoEntry : *transition) {
        // If there is a chooser this is the special case of a "may consumes"
        if (esConsumesInfoEntry.chooser_) {
          auto const& esTagGetterInfos = esConsumesInfoEntry.chooser_->tagGetter().lookup();

          for (auto const& esTagGetterInfo : esTagGetterInfos) {
            auto [componentDescription, produceMethodID] =
                iPI.componentAndProduceMethodID(esConsumesInfoEntry.recordKey_, esTagGetterInfo.index_);
            assert(componentDescription);
            if (alreadyFound.insert(componentDescription->id_).second) {
              esModules.push_back(componentDescription);
            }
          }

          // Handle cases not involving "may consumes"
        } else {
          auto [componentDescription, produceMethodID] =
              iPI.componentAndProduceMethodID(esConsumesInfoEntry.recordKey_, *itResolver);
          if (componentDescription) {
            if (alreadyFound.insert(componentDescription->id_).second) {
              esModules.push_back(componentDescription);
            }
          }
        }
        ++itResolver;
      }
      ++it;
    }
  }

  std::vector<std::vector<ESModuleConsumesInfo>> ESProducer::esModuleConsumesInfos(
      eventsetup::ESRecordsToProductResolverIndices const& iPI) const {
    std::vector<std::vector<ESModuleConsumesInfo>> result;
    result.resize(consumesInfos_.size());

    ESModuleConsumesInfo info;

    auto resultForTransition = result.begin();
    auto resolversForTransition = itemsToGetFromRecords_.begin();
    for (auto const& esConsumesInfo : consumesInfos_) {
      auto itResolver = resolversForTransition->begin();
      for (auto const& esConsumesInfoEntry : *esConsumesInfo) {
        info.eventSetupRecordType_ = esConsumesInfoEntry.recordKey_.name();
        info.productType_ = esConsumesInfoEntry.productKey_.type().name();
        info.moduleType_ = {};
        info.moduleLabel_ = {};
        info.produceMethodIDOfProducer_ = 0;
        info.isSource_ = false;
        info.isLooper_ = false;
        info.moduleLabelMismatch_ = false;

        // If there is a chooser this is the special case of a "may consumes"
        if (esConsumesInfoEntry.chooser_) {
          info.requestedModuleLabel_ = {};
          info.mayConsumes_ = true;
          info.mayConsumesFirstEntry_ = true;

          auto const& esTagGetterInfos = esConsumesInfoEntry.chooser_->tagGetter().lookup();

          if (esTagGetterInfos.empty()) {
            info.productLabel_ = {};
            info.mayConsumesNoProducts_ = true;
            resultForTransition->push_back(info);
          }

          // In the "may consumes" case, we iterate over all the possible data products
          // the EventSetup can produce with matching record type and product type.
          // With the current design of the mayConsumes feature, there is no way to
          // know in advance which productLabel or moduleLabel will be requested.
          // Maybe none will be. requestedModuleLabel and moduleLabelMismatch
          // are meaningless for "may consumes" cases.
          for (auto const& esTagGetterInfo : esTagGetterInfos) {
            info.productLabel_ = esTagGetterInfo.productLabel_;
            info.moduleLabel_ = esTagGetterInfo.moduleLabel_;
            info.mayConsumesNoProducts_ = false;

            auto [componentDescription, produceMethodID] =
                iPI.componentAndProduceMethodID(esConsumesInfoEntry.recordKey_, esTagGetterInfo.index_);
            assert(componentDescription);
            info.moduleType_ = componentDescription->type_;
            assert(info.moduleLabel_ ==
                   (componentDescription->label_.empty() ? componentDescription->type_ : componentDescription->label_));

            info.produceMethodIDOfProducer_ = produceMethodID;
            info.isSource_ = componentDescription->isSource_;
            info.isLooper_ = componentDescription->isLooper_;

            resultForTransition->push_back(info);

            info.mayConsumesFirstEntry_ = false;
          }

          // Handle cases not involving "may consumes"
        } else {
          info.productLabel_ = esConsumesInfoEntry.productKey_.name().value();
          info.requestedModuleLabel_ = esConsumesInfoEntry.moduleLabel_;
          info.moduleLabelMismatch_ = *itResolver == ESResolverIndex::moduleLabelDoesNotMatch();
          info.mayConsumes_ = false;
          info.mayConsumesFirstEntry_ = false;
          info.mayConsumesNoProducts_ = false;

          auto [componentDescription, produceMethodID] =
              iPI.componentAndProduceMethodID(esConsumesInfoEntry.recordKey_, *itResolver);

          if (componentDescription) {
            info.moduleType_ = componentDescription->type_;
            info.moduleLabel_ =
                componentDescription->label_.empty() ? componentDescription->type_ : componentDescription->label_;
            info.produceMethodIDOfProducer_ = produceMethodID;
            info.isSource_ = componentDescription->isSource_;
            info.isLooper_ = componentDescription->isLooper_;
          }
          resultForTransition->push_back(info);
        }
        ++itResolver;
      }
      ++resolversForTransition;
      ++resultForTransition;
      ++info.produceMethodIDOfConsumer_;
    }
    return result;
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
