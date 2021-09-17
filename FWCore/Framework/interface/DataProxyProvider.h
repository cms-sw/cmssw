#ifndef FWCore_Framework_DataProxyProvider_h
#define FWCore_Framework_DataProxyProvider_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyProvider
//
/**\class edm::eventsetup::DataProxyProvider

 Description: Lowest level base class for modules which produce
              data for the EventSetup system.

 Usage:

    In most cases, the methods in this class are used exclusively by the
    Framework. Usually, EventSetup modules producing data inherit from
    the ESProducer base class which inherits from this class. The ESProducer
    base class takes care of overriding the registerProxies function
    and calling usingRecord or usingRecordWithKey.

    In cases where the ESProducer base class is not used (PoolDBESSource/
    CondDBESSource is the main such class) then the registerProxies
    function must be overridden. For the same EventSetupRecordKey, the
    vector returned should contain the same DataKeys in the same order for
    all the different iovIndexes. DataProxies associated with the same
    EventSetupRecordKey should have caches that use different memory, but
    other than that they should also be the same.

    Classes that derive from this class should also call usingRecord
    or usingRecordWithKey in their constructor to announce the records
    they provide data for.

    All other functions are intended for use by the Framework or tests
    and should not be called in classes derived from DataProxyProvider.
    They are primarily used when initializing the EventSetup system
    so the DataProxies are available for use when needed.
*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 14:21:58 EST 2005
//

// system include files
#include <memory>
#include <set>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;

  namespace eventsetup {
    class DataProxy;
    class ESRecordsToProxyIndices;

    class DataProxyProvider {
    public:
      DataProxyProvider();
      DataProxyProvider(const DataProxyProvider&) = delete;
      const DataProxyProvider& operator=(const DataProxyProvider&) = delete;
      virtual ~DataProxyProvider() noexcept(false);

      class DataProxyContainer;

      class KeyedProxies {
      public:
        KeyedProxies(DataProxyContainer*, unsigned int recordIndex);

        bool unInitialized() const;

        EventSetupRecordKey const& recordKey() const;

        void insert(std::vector<std::pair<DataKey, std::shared_ptr<DataProxy>>>&&,
                    std::string const& appendToDataLabel);

        bool contains(DataKey const& dataKey) const;

        unsigned int size() const;

        // Not an STL iterator and cannot be used as one
        class Iterator {
        public:
          DataKey& dataKey() { return *dataKeysIter_; }
          DataProxy* dataProxy() { return dataProxiesIter_->get(); }
          Iterator& operator++();

          bool operator!=(Iterator const& right) const { return dataKeysIter_ != right.dataKeysIter_; }

          // Warning: dereference operator does not return a reference to an element in a container.
          // The return type is nonstandard because the iteration is simultaneous over 2 containers.
          // This return type is used in "ranged-based for" loops.
          struct KeyedProxy {
            KeyedProxy(DataKey& dataKey, DataProxy* dataProxy) : dataKey_(dataKey), dataProxy_(dataProxy) {}
            DataKey& dataKey_;
            DataProxy* dataProxy_;
          };
          KeyedProxy operator*() { return KeyedProxy(dataKey(), dataProxy()); }

        private:
          friend KeyedProxies;
          Iterator(std::vector<DataKey>::iterator dataKeysIter,
                   std::vector<edm::propagate_const<std::shared_ptr<DataProxy>>>::iterator dataProxiesIter);

          std::vector<DataKey>::iterator dataKeysIter_;
          std::vector<edm::propagate_const<std::shared_ptr<DataProxy>>>::iterator dataProxiesIter_;
        };

        Iterator begin();
        Iterator end();

      private:
        edm::propagate_const<DataProxyContainer*> dataProxyContainer_;
        unsigned int recordIndex_;
        unsigned int dataProxiesIndex_;
      };

      struct PerRecordInfo {
        PerRecordInfo(const EventSetupRecordKey&);
        bool operator<(const PerRecordInfo& right) const { return recordKey_ < right.recordKey_; }
        bool operator==(const PerRecordInfo& right) const { return recordKey_ == right.recordKey_; }

        EventSetupRecordKey recordKey_;
        unsigned int nDataKeys_ = 0;
        unsigned int indexToDataKeys_;
        unsigned int nIOVs_ = 0;
        unsigned int indexToKeyedProxies_ = 0;
      };

      class DataProxyContainer {
      public:
        void usingRecordWithKey(const EventSetupRecordKey&);
        bool isUsingRecord(const EventSetupRecordKey&) const;
        std::set<EventSetupRecordKey> usingRecords() const;
        void fillRecordsNotAllowingConcurrentIOVs(std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const;

        void sortEventSetupRecordKeys();
        void createKeyedProxies(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs);

        KeyedProxies& keyedProxies(const EventSetupRecordKey& iRecordKey, unsigned int iovIndex);

      private:
        friend KeyedProxies;

        std::vector<PerRecordInfo> perRecordInfos_;
        std::vector<KeyedProxies> keyedProxiesCollection_;
        std::vector<DataKey> dataKeys_;
        std::vector<edm::propagate_const<std::shared_ptr<DataProxy>>> dataProxies_;
      };

      bool isUsingRecord(const EventSetupRecordKey& key) const { return dataProxyContainer_.isUsingRecord(key); }
      std::set<EventSetupRecordKey> usingRecords() const { return dataProxyContainer_.usingRecords(); }
      void fillRecordsNotAllowingConcurrentIOVs(std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const {
        dataProxyContainer_.fillRecordsNotAllowingConcurrentIOVs(recordsNotAllowingConcurrentIOVs);
      }

      virtual void initConcurrentIOVs(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs) {}

      void createKeyedProxies(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs) {
        dataProxyContainer_.createKeyedProxies(key, nConcurrentIOVs);
        initConcurrentIOVs(key, nConcurrentIOVs);
      }

      const ComponentDescription& description() const { return description_; }

      virtual void updateLookup(ESRecordsToProxyIndices const&);

      void setDescription(const ComponentDescription& iDescription) { description_ = iDescription; }

      /**This method is only to be called by the framework, it sets the string
        which will be appended to the labels of all data products being produced
      **/
      void setAppendToDataLabel(const edm::ParameterSet&);

      KeyedProxies& keyedProxies(const EventSetupRecordKey& iRecordKey, unsigned int iovIndex = 0);

      /**Used to add parameters available to all inheriting classes
      */
      static void prevalidate(ConfigurationDescriptions&);

    protected:
      template <class T>
      void usingRecord() {
        usingRecordWithKey(EventSetupRecordKey::makeKey<T>());
      }

      void usingRecordWithKey(const EventSetupRecordKey& key) { dataProxyContainer_.usingRecordWithKey(key); }

      using KeyedProxiesVector = std::vector<std::pair<DataKey, std::shared_ptr<DataProxy>>>;
      virtual KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int iovIndex) = 0;

    private:
      // ---------- member data --------------------------------
      DataProxyContainer dataProxyContainer_;
      ComponentDescription description_;
      std::string appendToDataLabel_;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
