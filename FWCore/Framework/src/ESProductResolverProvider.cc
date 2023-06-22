// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverProvider
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Mon Mar 28 15:07:54 EST 2005
//

// system include files
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

// user include files
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

constexpr auto kInvalidIndex = std::numeric_limits<unsigned int>::max();

namespace edm {
  namespace eventsetup {

    ESProductResolverProvider::ESProductResolverProvider() {}

    ESProductResolverProvider::~ESProductResolverProvider() noexcept(false) {}

    ESProductResolverProvider::KeyedResolvers::KeyedResolvers(ESProductResolverContainer* productResolverContainer, unsigned int recordIndex)
        : productResolverContainer_(productResolverContainer), recordIndex_(recordIndex), productResolversIndex_(kInvalidIndex) {}

    bool ESProductResolverProvider::KeyedResolvers::unInitialized() const { return productResolversIndex_ == kInvalidIndex; }

    EventSetupRecordKey const& ESProductResolverProvider::KeyedResolvers::recordKey() const {
      return productResolverContainer_->perRecordInfos_[recordIndex_].recordKey_;
    }

    void ESProductResolverProvider::KeyedResolvers::insert(std::vector<std::pair<DataKey, std::shared_ptr<ESProductResolver>>>&& proxies,
                                                 std::string const& appendToDataLabel) {
      PerRecordInfo& perRecordInfo = productResolverContainer_->perRecordInfos_[recordIndex_];
      if (perRecordInfo.indexToDataKeys_ == kInvalidIndex) {
        perRecordInfo.nDataKeys_ = proxies.size();
        perRecordInfo.indexToDataKeys_ = productResolverContainer_->dataKeys_.size();
        for (auto const& it : proxies) {
          productResolverContainer_->dataKeys_.push_back(it.first);
        }
      } else {
        assert(perRecordInfo.nDataKeys_ == proxies.size());
        unsigned index = 0;
        for (auto const& it : proxies) {
          if (appendToDataLabel.empty()) {
            assert(it.first == productResolverContainer_->dataKeys_[perRecordInfo.indexToDataKeys_ + index]);
          } else {
            assert(it.first.type() == productResolverContainer_->dataKeys_[perRecordInfo.indexToDataKeys_ + index].type());
            auto lengthDataLabel = std::strlen(it.first.name().value());
            assert(std::strncmp(it.first.name().value(),
                                productResolverContainer_->dataKeys_[perRecordInfo.indexToDataKeys_ + index].name().value(),
                                lengthDataLabel) == 0);
          }
          ++index;
        }
      }
      assert(unInitialized());
      productResolversIndex_ = productResolverContainer_->productResolvers_.size();
      for (auto const& it : proxies) {
        productResolverContainer_->productResolvers_.emplace_back(it.second);
      }
    }

    bool ESProductResolverProvider::KeyedResolvers::contains(DataKey const& dataKey) const {
      PerRecordInfo const& perRecordInfo = productResolverContainer_->perRecordInfos_[recordIndex_];
      auto iter = productResolverContainer_->dataKeys_.begin() + perRecordInfo.indexToDataKeys_;
      auto iterEnd = iter + perRecordInfo.nDataKeys_;
      for (; iter != iterEnd; ++iter) {
        if (*iter == dataKey) {
          return true;
        }
      }
      return false;
    }

    unsigned int ESProductResolverProvider::KeyedResolvers::size() const {
      return productResolverContainer_->perRecordInfos_[recordIndex_].nDataKeys_;
    }

    ESProductResolverProvider::KeyedResolvers::Iterator& ESProductResolverProvider::KeyedResolvers::Iterator::operator++() {
      ++dataKeysIter_;
      ++productResolversIter_;
      return *this;
    }

    ESProductResolverProvider::KeyedResolvers::Iterator::Iterator(
        std::vector<DataKey>::iterator dataKeysIter,
        std::vector<edm::propagate_const<std::shared_ptr<ESProductResolver>>>::iterator productResolversIter)
        : dataKeysIter_(dataKeysIter), productResolversIter_(productResolversIter) {}

    ESProductResolverProvider::KeyedResolvers::Iterator ESProductResolverProvider::KeyedResolvers::begin() {
      return Iterator(
          productResolverContainer_->dataKeys_.begin() + productResolverContainer_->perRecordInfos_[recordIndex_].indexToDataKeys_,
          productResolverContainer_->productResolvers_.begin() + productResolversIndex_);
    }

    ESProductResolverProvider::KeyedResolvers::Iterator ESProductResolverProvider::KeyedResolvers::end() {
      unsigned int nDataKeys = productResolverContainer_->perRecordInfos_[recordIndex_].nDataKeys_;
      return Iterator(productResolverContainer_->dataKeys_.begin() +
                          productResolverContainer_->perRecordInfos_[recordIndex_].indexToDataKeys_ + nDataKeys,
                      productResolverContainer_->productResolvers_.begin() + productResolversIndex_ + nDataKeys);
    }

    ESProductResolverProvider::PerRecordInfo::PerRecordInfo(const EventSetupRecordKey& key)
        : recordKey_(key), indexToDataKeys_(kInvalidIndex) {}

    void ESProductResolverProvider::ESProductResolverContainer::usingRecordWithKey(const EventSetupRecordKey& iKey) {
      assert(keyedResolversCollection_.empty());
      perRecordInfos_.emplace_back(iKey);
    }

    bool ESProductResolverProvider::ESProductResolverContainer::isUsingRecord(const EventSetupRecordKey& iKey) const {
      auto lb = std::lower_bound(perRecordInfos_.begin(), perRecordInfos_.end(), PerRecordInfo(iKey));
      return (lb != perRecordInfos_.end() && iKey == lb->recordKey_);
    }

    std::set<EventSetupRecordKey> ESProductResolverProvider::ESProductResolverContainer::usingRecords() const {
      std::set<EventSetupRecordKey> returnValue;
      for (auto const& it : perRecordInfos_) {
        returnValue.insert(returnValue.end(), it.recordKey_);
      }
      return returnValue;
    }

    void ESProductResolverProvider::ESProductResolverContainer::fillRecordsNotAllowingConcurrentIOVs(
        std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const {
      for (auto const& it : perRecordInfos_) {
        const EventSetupRecordKey& key = it.recordKey_;
        if (!allowConcurrentIOVs(key)) {
          recordsNotAllowingConcurrentIOVs.insert(recordsNotAllowingConcurrentIOVs.end(), key);
        }
      }
    }

    void ESProductResolverProvider::ESProductResolverContainer::sortEventSetupRecordKeys() {
      std::sort(perRecordInfos_.begin(), perRecordInfos_.end());
      perRecordInfos_.erase(std::unique(perRecordInfos_.begin(), perRecordInfos_.end()), perRecordInfos_.end());
    }

    void ESProductResolverProvider::ESProductResolverContainer::createKeyedResolvers(EventSetupRecordKey const& key,
                                                                   unsigned int nConcurrentIOVs) {
      if (keyedResolversCollection_.empty()) {
        sortEventSetupRecordKeys();
      }
      assert(nConcurrentIOVs > 0U);
      auto lb = std::lower_bound(perRecordInfos_.begin(), perRecordInfos_.end(), PerRecordInfo(key));
      assert(lb != perRecordInfos_.end() && key == lb->recordKey_);
      if (lb->nIOVs_ == 0) {
        lb->nIOVs_ = nConcurrentIOVs;
        auto recordIndex = std::distance(perRecordInfos_.begin(), lb);
        lb->indexToKeyedResolvers_ = keyedResolversCollection_.size();
        for (unsigned int i = 0; i < nConcurrentIOVs; ++i) {
          keyedResolversCollection_.emplace_back(this, recordIndex);
        }
      }
    }

    ESProductResolverProvider::KeyedResolvers& ESProductResolverProvider::ESProductResolverContainer::keyedResolvers(
        const EventSetupRecordKey& iRecordKey, unsigned int iovIndex) {
      auto lb = std::lower_bound(perRecordInfos_.begin(), perRecordInfos_.end(), PerRecordInfo(iRecordKey));
      assert(lb != perRecordInfos_.end() && iRecordKey == lb->recordKey_);
      assert(iovIndex < lb->nIOVs_);
      return keyedResolversCollection_[lb->indexToKeyedResolvers_ + iovIndex];
    }

    void ESProductResolverProvider::updateLookup(eventsetup::ESRecordsToProductResolverIndices const&) {}

    void ESProductResolverProvider::setAppendToDataLabel(const edm::ParameterSet& iToAppend) {
      std::string oldValue(appendToDataLabel_);
      //this can only be changed once and the default value is the empty string
      assert(oldValue.empty());

      const std::string kParamName("appendToDataLabel");
      if (iToAppend.exists(kParamName)) {
        appendToDataLabel_ = iToAppend.getParameter<std::string>(kParamName);
      }
    }

    ESProductResolverProvider::KeyedResolvers& ESProductResolverProvider::keyedResolvers(const EventSetupRecordKey& iRecordKey,
                                                                     unsigned int iovIndex) {
      KeyedResolvers& keyedResolvers = productResolverContainer_.keyedResolvers(iRecordKey, iovIndex);

      if (keyedResolvers.unInitialized()) {
        //delayed registration
        std::vector<std::pair<DataKey, std::shared_ptr<ESProductResolver>>> keyedResolversVector =
            registerResolvers(iRecordKey, iovIndex);
        keyedResolvers.insert(std::move(keyedResolversVector), appendToDataLabel_);

        bool mustChangeLabels = (!appendToDataLabel_.empty());
        for (auto keyedResolver : keyedResolvers) {
          keyedResolver.productResolver_->setProviderDescription(&description());
          if (mustChangeLabels) {
            //Using swap is fine since
            // 1) the data structure is not a map and so we have not sorted on the keys
            // 2) this is the first time filling this so no outside agency has yet seen
            //   the label and therefore can not be dependent upon its value
            std::string temp(std::string(keyedResolver.dataKey_.name().value()) + appendToDataLabel_);
            DataKey newKey(keyedResolver.dataKey_.type(), temp.c_str());
            swap(keyedResolver.dataKey_, newKey);
          }
        }
      }
      return keyedResolvers;
    }

    static const std::string kAppendToDataLabel("appendToDataLabel");

    void ESProductResolverProvider::prevalidate(ConfigurationDescriptions& iDesc) {
      if (iDesc.defaultDescription()) {
        if (iDesc.defaultDescription()->isLabelUnused(kAppendToDataLabel)) {
          iDesc.defaultDescription()->add<std::string>(kAppendToDataLabel, std::string(""));
        }
      }
      for (auto& v : iDesc) {
        if (v.second.isLabelUnused(kAppendToDataLabel)) {
          v.second.add<std::string>(kAppendToDataLabel, std::string(""));
        }
      }
    }

  }  // namespace eventsetup
}  // namespace edm
