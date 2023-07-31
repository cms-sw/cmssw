// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     CondHDF5ESSource
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 16 Jun 2023 15:17:53 GMT
//

// system include files
#include <cassert>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include "CondFormats/SerializationHelper/interface/SerializationHelperFactory.h"

#include "IOVSyncValue.h"
#include "DataProduct.h"
#include "Record.h"
#include "HDF5DataProxy.h"
#include "convertSyncValue.h"
#include "h5_File.h"
#include "h5_Group.h"
#include "h5_DataSet.h"
#include "h5_Attribute.h"

using namespace cond::hdf5;

class CondHDF5ESSource : public edm::EventSetupRecordIntervalFinder, public edm::eventsetup::DataProxyProvider {
public:
  using EventSetupRecordKey = edm::eventsetup::EventSetupRecordKey;
  explicit CondHDF5ESSource(edm::ParameterSet const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool isConcurrentFinder() const final { return true; }
  void setIntervalFor(EventSetupRecordKey const&, edm::IOVSyncValue const&, edm::ValidityInterval&) final;
  KeyedProxiesVector registerProxies(EventSetupRecordKey const&, unsigned int iovIndex) final;

  edm::SerialTaskQueue queue_;
  std::mutex mutex_;
  std::vector<Record> records_;
  std::string filename_;
  cms::h5::File file_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CondHDF5ESSource::CondHDF5ESSource(edm::ParameterSet const& iPSet)
    : filename_(iPSet.getUntrackedParameter<std::string>("filename")), file_(filename_, cms::h5::File::kReadOnly) {
  const auto globalTagsGroup = file_.findGroup("GlobalTags");
  const auto chosenTag = globalTagsGroup->findGroup(iPSet.getParameter<std::string>("globalTag"));
  const auto tagsDataSet = chosenTag->findDataSet("Tags");
  const auto recordsGroup = file_.findGroup("Records");

  std::vector<hobj_ref_t> tags = tagsDataSet->readRefs();

  std::set<std::string> recordsToExclude;
  {
    auto exclude = iPSet.getParameter<std::vector<std::string>>("excludeRecords");
    recordsToExclude = std::set(exclude.begin(), exclude.end());
  }

  for (auto t : tags) {
    auto tagGroup = file_.derefGroup(t);
    Record record;
    record.name_ = tagGroup->findAttribute("record")->readString();
    //std::cout << record.name_ << std::endl;

    if (recordsToExclude.end() != recordsToExclude.find(record.name_)) {
      std::cout << "excluding " << record.name_ << std::endl;
      continue;
    }

    auto recordGroup = recordsGroup->findGroup(record.name_);
    //std::cout << "found record group" << std::endl;
    auto dataProductsGroup = recordGroup->findGroup("DataProducts");
    //std::cout << "found DataProducts group" << std::endl;

    for (size_t i = 0; i < dataProductsGroup->getNumObjs(); ++i) {
      std::string productGroupName = dataProductsGroup->getObjnameByIdx(i);
      //std::cout << "looking for " << productGroupName << std::endl;
      auto dataProductGroup = dataProductsGroup->findGroup(productGroupName);

      auto const typeAttr = dataProductGroup->findAttribute("type");
      std::string typeName = typeAttr->readString();
      //loading the factory should also trigger registering the Record and DataProduct keys
      cond::serialization::SerializationHelperFactory::get()->create(typeName);
      std::string name = productGroupName.substr(typeName.size() + 1, productGroupName.size());
      if (name.size() == 1 and name[0] == '-') {
        name = std::string();
      }
      record.dataProducts_.emplace_back(std::move(name), std::move(typeName));
    }

    {
      auto const typeAttr = tagGroup->findAttribute("time_type");
      std::string typeName = typeAttr->readString();
      record.iovIsRunLumi_ = (typeName == "run_lumi");
    }

    std::vector<hobj_ref_t> payloadRefForIOVs;
    {
      auto const firstDataSet = tagGroup->findDataSet("first");
      auto const lastDataSet = tagGroup->findDataSet("last");

      record.iovFirsts_ = firstDataSet->readSyncValues();
      record.iovLasts_ = lastDataSet->readSyncValues();

      {
        auto const payloadDataSet = tagGroup->findDataSet("payload");
        payloadRefForIOVs = payloadDataSet->readRefs();
        assert(payloadRefForIOVs.size() == record.iovFirsts_.size() * record.dataProducts_.size());
      }
    }
    size_t dataProductIndex = 0;
    for (auto r : payloadRefForIOVs) {
      record.dataProducts_[dataProductIndex].payloadForIOVs_.push_back(r);
      ++dataProductIndex;
      if (dataProductIndex >= record.dataProducts_.size()) {
        dataProductIndex = 0;
      }
    }

    //now that we've loaded a plugin that is associated to the record, the record should be registered
    auto key =
        edm::eventsetup::EventSetupRecordKey(edm::eventsetup::heterocontainer::HCTypeTag::findType(record.name_));
    assert(key != edm::eventsetup::heterocontainer::HCTypeTag());
    //tell system we retrieve this Record
    findingRecordWithKey(key);
    usingRecordWithKey(key);

    records_.emplace_back(std::move(record));
  }
  std::sort(records_.begin(), records_.end(), [](auto const& l, auto const& r) { return l.name_ < r.name_; });
}

void CondHDF5ESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("filename")->setComment("HDF5 file containing the conditions");
  desc.add<std::string>("globalTag")->setComment("Which global tag to use from the file");
  desc.add<std::vector<std::string>>("excludeRecords", std::vector<std::string>())
      ->setComment("List of Records that should not be read from the file");

  descriptions.addDefault(desc);
}

void CondHDF5ESSource::setIntervalFor(EventSetupRecordKey const& iRecordKey,
                                      edm::IOVSyncValue const& iSync,
                                      edm::ValidityInterval& iIOV) {
  using namespace cond::hdf5;

  auto const itRecord =
      std::lower_bound(records_.begin(), records_.end(), iRecordKey.name(), [](auto const& iE, auto const& iV) {
        return iE.name_ < iV;
      });
  assert(itRecord != records_.end());
  auto const& record = *itRecord;
  assert(record.name_ == iRecordKey.name());
  auto sync = convertSyncValue(iSync, record.iovIsRunLumi_);
  auto itFound = findMatchingFirst(record.iovFirsts_, sync);
  if (itFound == record.iovFirsts_.end()) {
    //std::cout << "BAD SYNC for record " << iRecordKey.name() << std::endl;
    iIOV = edm::ValidityInterval::invalidInterval();
    return;
  }
  iIOV = edm::ValidityInterval{
      convertSyncValue(*itFound, record.iovIsRunLumi_),
      convertSyncValue(record.iovLasts_[itFound - record.iovFirsts_.begin()], record.iovIsRunLumi_)};
}

CondHDF5ESSource::KeyedProxiesVector CondHDF5ESSource::registerProxies(EventSetupRecordKey const& iRecordKey,
                                                                       unsigned int iovIndex) {
  CondHDF5ESSource::KeyedProxiesVector returnValue;

  //std::cout << "Register proxies called " << iRecordKey.name() << std::endl;
  auto const itRecord =
      std::lower_bound(records_.begin(), records_.end(), iRecordKey.name(), [](auto const& iE, auto const& iV) {
        return iE.name_ < iV;
      });
  assert(itRecord != records_.end());
  auto const& record = *itRecord;
  assert(record.name_ == iRecordKey.name());
  for (auto const& dataProduct : record.dataProducts_) {
    //std::cout << "Making DataProduct " << dataProduct.type_ << " '" << dataProduct.name_ << "' for Record "
    //          << record.name_ << std::endl;
    auto helper = cond::serialization::SerializationHelperFactory::get()->create(dataProduct.type_);
    returnValue.emplace_back(
        edm::eventsetup::DataKey(edm::eventsetup::heterocontainer::HCTypeTag::findType(dataProduct.type_),
                                 dataProduct.name_.c_str()),
        std::make_shared<HDF5DataProxy>(&queue_, &mutex_, std::move(helper), &file_, filename_, &record, &dataProduct));
  }
  return returnValue;
}

DEFINE_FWK_EVENTSETUP_SOURCE(CondHDF5ESSource);
