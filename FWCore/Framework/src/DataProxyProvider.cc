// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyProvider
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
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

constexpr auto kInvalidIndex = std::numeric_limits<unsigned int>::max();

namespace edm {
  namespace eventsetup {

    DataProxyProvider::DataProxyProvider() {}

    DataProxyProvider::~DataProxyProvider() noexcept(false) {}

    DataProxyProvider::KeyedProxies::KeyedProxies(DataProxyContainer* dataProxyContainer, unsigned int recordIndex)
        : dataProxyContainer_(dataProxyContainer), recordIndex_(recordIndex), dataProxiesIndex_(kInvalidIndex) {}

    bool DataProxyProvider::KeyedProxies::unInitialized() const { return dataProxiesIndex_ == kInvalidIndex; }

    EventSetupRecordKey const& DataProxyProvider::KeyedProxies::recordKey() const {
      return dataProxyContainer_->perRecordInfos_[recordIndex_].recordKey_;
    }

    void DataProxyProvider::KeyedProxies::insert(std::vector<std::pair<DataKey, std::shared_ptr<DataProxy>>>&& proxies,
                                                 std::string const& appendToDataLabel) {
      PerRecordInfo& perRecordInfo = dataProxyContainer_->perRecordInfos_[recordIndex_];
      if (perRecordInfo.indexToDataKeys_ == kInvalidIndex) {
        perRecordInfo.nDataKeys_ = proxies.size();
        perRecordInfo.indexToDataKeys_ = dataProxyContainer_->dataKeys_.size();
        for (auto const& it : proxies) {
          dataProxyContainer_->dataKeys_.push_back(it.first);
        }
      } else {
        assert(perRecordInfo.nDataKeys_ == proxies.size());
        unsigned index = 0;
        for (auto const& it : proxies) {
          if (appendToDataLabel.empty()) {
            assert(it.first == dataProxyContainer_->dataKeys_[perRecordInfo.indexToDataKeys_ + index]);
          } else {
            assert(it.first.type() == dataProxyContainer_->dataKeys_[perRecordInfo.indexToDataKeys_ + index].type());
            auto lengthDataLabel = std::strlen(it.first.name().value());
            assert(std::strncmp(it.first.name().value(),
                                dataProxyContainer_->dataKeys_[perRecordInfo.indexToDataKeys_ + index].name().value(),
                                lengthDataLabel) == 0);
          }
          ++index;
        }
      }
      assert(unInitialized());
      dataProxiesIndex_ = dataProxyContainer_->dataProxies_.size();
      for (auto const& it : proxies) {
        dataProxyContainer_->dataProxies_.emplace_back(it.second);
      }
    }

    bool DataProxyProvider::KeyedProxies::contains(DataKey const& dataKey) const {
      PerRecordInfo const& perRecordInfo = dataProxyContainer_->perRecordInfos_[recordIndex_];
      auto iter = dataProxyContainer_->dataKeys_.begin() + perRecordInfo.indexToDataKeys_;
      auto iterEnd = iter + perRecordInfo.nDataKeys_;
      for (; iter != iterEnd; ++iter) {
        if (*iter == dataKey) {
          return true;
        }
      }
      return false;
    }

    unsigned int DataProxyProvider::KeyedProxies::size() const {
      return dataProxyContainer_->perRecordInfos_[recordIndex_].nDataKeys_;
    }

    DataProxyProvider::KeyedProxies::Iterator& DataProxyProvider::KeyedProxies::Iterator::operator++() {
      ++dataKeysIter_;
      ++dataProxiesIter_;
      return *this;
    }

    DataProxyProvider::KeyedProxies::Iterator::Iterator(
        std::vector<DataKey>::iterator dataKeysIter,
        std::vector<edm::propagate_const<std::shared_ptr<DataProxy>>>::iterator dataProxiesIter)
        : dataKeysIter_(dataKeysIter), dataProxiesIter_(dataProxiesIter) {}

    DataProxyProvider::KeyedProxies::Iterator DataProxyProvider::KeyedProxies::begin() {
      return Iterator(
          dataProxyContainer_->dataKeys_.begin() + dataProxyContainer_->perRecordInfos_[recordIndex_].indexToDataKeys_,
          dataProxyContainer_->dataProxies_.begin() + dataProxiesIndex_);
    }

    DataProxyProvider::KeyedProxies::Iterator DataProxyProvider::KeyedProxies::end() {
      unsigned int nDataKeys = dataProxyContainer_->perRecordInfos_[recordIndex_].nDataKeys_;
      return Iterator(dataProxyContainer_->dataKeys_.begin() +
                          dataProxyContainer_->perRecordInfos_[recordIndex_].indexToDataKeys_ + nDataKeys,
                      dataProxyContainer_->dataProxies_.begin() + dataProxiesIndex_ + nDataKeys);
    }

    DataProxyProvider::PerRecordInfo::PerRecordInfo(const EventSetupRecordKey& key)
        : recordKey_(key), indexToDataKeys_(kInvalidIndex) {}

    void DataProxyProvider::DataProxyContainer::usingRecordWithKey(const EventSetupRecordKey& iKey) {
      assert(keyedProxiesCollection_.empty());
      perRecordInfos_.emplace_back(iKey);
    }

    bool DataProxyProvider::DataProxyContainer::isUsingRecord(const EventSetupRecordKey& iKey) const {
      auto lb = std::lower_bound(perRecordInfos_.begin(), perRecordInfos_.end(), PerRecordInfo(iKey));
      return (lb != perRecordInfos_.end() && iKey == lb->recordKey_);
    }

    std::set<EventSetupRecordKey> DataProxyProvider::DataProxyContainer::usingRecords() const {
      std::set<EventSetupRecordKey> returnValue;
      for (auto const& it : perRecordInfos_) {
        returnValue.insert(returnValue.end(), it.recordKey_);
      }
      return returnValue;
    }

    void DataProxyProvider::DataProxyContainer::fillRecordsNotAllowingConcurrentIOVs(
        std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const {
      for (auto const& it : perRecordInfos_) {
        const EventSetupRecordKey& key = it.recordKey_;
        if (!allowConcurrentIOVs(key)) {
          recordsNotAllowingConcurrentIOVs.insert(recordsNotAllowingConcurrentIOVs.end(), key);
        }
      }
    }

    void DataProxyProvider::DataProxyContainer::sortEventSetupRecordKeys() {
      std::sort(perRecordInfos_.begin(), perRecordInfos_.end());
      perRecordInfos_.erase(std::unique(perRecordInfos_.begin(), perRecordInfos_.end()), perRecordInfos_.end());
    }

    void DataProxyProvider::DataProxyContainer::createKeyedProxies(EventSetupRecordKey const& key,
                                                                   unsigned int nConcurrentIOVs) {
      if (keyedProxiesCollection_.empty()) {
        sortEventSetupRecordKeys();
      }
      assert(nConcurrentIOVs > 0U);
      auto lb = std::lower_bound(perRecordInfos_.begin(), perRecordInfos_.end(), PerRecordInfo(key));
      assert(lb != perRecordInfos_.end() && key == lb->recordKey_);
      if (lb->nIOVs_ == 0) {
        lb->nIOVs_ = nConcurrentIOVs;
        auto recordIndex = std::distance(perRecordInfos_.begin(), lb);
        lb->indexToKeyedProxies_ = keyedProxiesCollection_.size();
        for (unsigned int i = 0; i < nConcurrentIOVs; ++i) {
          keyedProxiesCollection_.emplace_back(this, recordIndex);
        }
      }
    }

    DataProxyProvider::KeyedProxies& DataProxyProvider::DataProxyContainer::keyedProxies(
        const EventSetupRecordKey& iRecordKey, unsigned int iovIndex) {
      auto lb = std::lower_bound(perRecordInfos_.begin(), perRecordInfos_.end(), PerRecordInfo(iRecordKey));
      assert(lb != perRecordInfos_.end() && iRecordKey == lb->recordKey_);
      assert(iovIndex < lb->nIOVs_);
      return keyedProxiesCollection_[lb->indexToKeyedProxies_ + iovIndex];
    }

    void DataProxyProvider::updateLookup(eventsetup::ESRecordsToProxyIndices const&) {}

    void DataProxyProvider::setAppendToDataLabel(const edm::ParameterSet& iToAppend) {
      std::string oldValue(appendToDataLabel_);
      //this can only be changed once and the default value is the empty string
      assert(oldValue.empty());

      const std::string kParamName("appendToDataLabel");
      if (iToAppend.exists(kParamName)) {
        appendToDataLabel_ = iToAppend.getParameter<std::string>(kParamName);
      }
    }

    DataProxyProvider::KeyedProxies& DataProxyProvider::keyedProxies(const EventSetupRecordKey& iRecordKey,
                                                                     unsigned int iovIndex) {
      KeyedProxies& keyedProxies = dataProxyContainer_.keyedProxies(iRecordKey, iovIndex);

      if (keyedProxies.unInitialized()) {
        //delayed registration
        std::vector<std::pair<DataKey, std::shared_ptr<DataProxy>>> keyedProxiesVector =
            registerProxies(iRecordKey, iovIndex);
        keyedProxies.insert(std::move(keyedProxiesVector), appendToDataLabel_);

        bool mustChangeLabels = (!appendToDataLabel_.empty());
        for (auto keyedProxy : keyedProxies) {
          keyedProxy.dataProxy_->setProviderDescription(&description());
          if (mustChangeLabels) {
            //Using swap is fine since
            // 1) the data structure is not a map and so we have not sorted on the keys
            // 2) this is the first time filling this so no outside agency has yet seen
            //   the label and therefore can not be dependent upon its value
            std::string temp(std::string(keyedProxy.dataKey_.name().value()) + appendToDataLabel_);
            DataKey newKey(keyedProxy.dataKey_.type(), temp.c_str());
            swap(keyedProxy.dataKey_, newKey);
          }
        }
      }
      return keyedProxies;
    }

    static const std::string kAppendToDataLabel("appendToDataLabel");

    void DataProxyProvider::prevalidate(ConfigurationDescriptions& iDesc) {
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
