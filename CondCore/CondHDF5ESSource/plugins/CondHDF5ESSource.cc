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
#include <H5Cpp.h>
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

#include "CondFormats/SerializationHelper/interface/SerializationHelperFactory.h"

#include "IOVSyncValue.h"
#include "DataProduct.h"
#include "Record.h"
#include "HDF5DataProxy.h"
#include "convertSyncValue.h"

namespace {
  inline hid_t string_type() {
    auto atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, H5T_VARIABLE);
    H5Tset_cset(atype, H5T_CSET_UTF8);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);
    return atype;
  }
}  // namespace

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

  std::vector<Record> records_;
  std::string filename_;
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
    : filename_(iPSet.getUntrackedParameter<std::string>("filename")) {
  H5::H5File file(filename_, H5F_ACC_RDONLY);

  const auto globalTagsGroup = file.openGroup("GlobalTags");
  const auto chosenTag = globalTagsGroup.openGroup(iPSet.getParameter<std::string>("globalTag"));
  const auto tagsDataSet = chosenTag.openDataSet("Tags");
  const auto recordsGroup = file.openGroup("Records");

  std::vector<hobj_ref_t> tags;
  {
    const auto tagsDataSpace = tagsDataSet.getSpace();
    assert(tagsDataSpace.isSimple());
    assert(tagsDataSpace.getSimpleExtentNdims() == 1);
    hsize_t size[1];
    tagsDataSpace.getSimpleExtentDims(size);
    tags.resize(size[0]);
  }

  tagsDataSet.read(&tags[0], H5::PredType::STD_REF_OBJ);

  for (auto t : tags) {
    assert(file.getRefObjType(&t) == H5O_TYPE_GROUP);
    H5::Group tagGroup(file, &t);
    std::string n = tagGroup.getObjName();
    std::cout << n << std::endl;
    std::cout << "prefix: '" << n.substr(0, 9) << "'" << std::endl;
    Record record;

    assert(n.substr(0, 9) == "/Records/");
    auto index = n.find('/', 10);
    record.name_ = n.substr(9, index - 9);
    std::cout << record.name_ << std::endl;

    H5::Group recordGroup = recordsGroup.openGroup(record.name_);
    std::cout << "found record group" << std::endl;
    H5::Group dataProductsGroup = recordGroup.openGroup("DataProducts");
    std::cout << "found DataProducts group" << std::endl;

    for (size_t i = 0; i < dataProductsGroup.getNumObjs(); ++i) {
      std::string productGroupName = dataProductsGroup.getObjnameByIdx(i);
      std::cout << "looking for " << productGroupName << std::endl;
      H5::Group dataProductGroup = dataProductsGroup.openGroup(productGroupName);

      auto const typeAttr = dataProductGroup.openAttribute("type");
      H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
      std::string typeName;
      typeAttr.read(str_type, typeName);
      //loading the factory should also trigger registering the Record and DataProduct keys
      cond::serialization::SerializationHelperFactory::get()->create(typeName);
      std::string name = productGroupName.substr(typeName.size() + 1, productGroupName.size());
      if (name.size() == 1 and name[0] == '-') {
        name = std::string();
      }
      record.dataProducts_.emplace_back(std::move(name), std::move(typeName));
    }

    {
      auto const typeAttr = tagGroup.openAttribute("time_type");
      H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
      std::string typeName;
      typeAttr.read(str_type, typeName);
      record.iovIsRunLumi_ = (typeName == "run_lumi");
    }

    std::vector<hobj_ref_t> payloadRefForIOVs;
    {
      auto const firstDataSet = tagGroup.openDataSet("first");
      auto const lastDataSet = tagGroup.openDataSet("last");
      assert(firstDataSet.getSpace().isSimple());
      assert(lastDataSet.getSpace().isSimple());
      assert(firstDataSet.getSpace().getSimpleExtentNdims() == 1);
      assert(lastDataSet.getSpace().getSimpleExtentNdims() == 1);

      hsize_t size[1];
      firstDataSet.getSpace().getSimpleExtentDims(size);
      {
        hsize_t sizeL[1];
        lastDataSet.getSpace().getSimpleExtentDims(sizeL);
        assert(sizeL[0] == size[0]);
      }

      H5::CompType syncValueType(sizeof(IOVSyncValue));
      syncValueType.insertMember("high", HOFFSET(IOVSyncValue, high_), H5::PredType::NATIVE_UINT32);
      syncValueType.insertMember("low", HOFFSET(IOVSyncValue, low_), H5::PredType::NATIVE_UINT32);

      record.iovFirsts_.resize(size[0]);
      record.iovLasts_.resize(size[0]);

      firstDataSet.read(&record.iovFirsts_[0], syncValueType);
      lastDataSet.read(&record.iovLasts_[0], syncValueType);

      {
        hsize_t refSize[2];
        auto const payloadDataSet = tagGroup.openDataSet("payload");
        payloadDataSet.getSpace().getSimpleExtentDims(refSize);
        assert(refSize[0] == record.iovFirsts_.size());
        assert(refSize[1] == record.dataProducts_.size());
        payloadRefForIOVs.resize(refSize[0] * refSize[1]);

        payloadDataSet.read(&payloadRefForIOVs[0], H5::PredType::STD_REF_OBJ);
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

  std::cout << "Register proxies called " << iRecordKey.name() << std::endl;
  auto const itRecord =
      std::lower_bound(records_.begin(), records_.end(), iRecordKey.name(), [](auto const& iE, auto const& iV) {
        return iE.name_ < iV;
      });
  assert(itRecord != records_.end());
  auto const& record = *itRecord;
  assert(record.name_ == iRecordKey.name());
  for (auto const& dataProduct : record.dataProducts_) {
    std::cout << "Making DataProduct " << dataProduct.type_ << " '" << dataProduct.name_ << "' for Record "
              << record.name_ << std::endl;
    auto helper = cond::serialization::SerializationHelperFactory::get()->create(dataProduct.type_);
    returnValue.emplace_back(
        edm::eventsetup::DataKey(edm::eventsetup::heterocontainer::HCTypeTag::findType(dataProduct.type_),
                                 dataProduct.name_.c_str()),
        std::make_shared<HDF5DataProxy>(std::move(helper), filename_, &record, &dataProduct));
  }
  return returnValue;
}

DEFINE_FWK_EVENTSETUP_SOURCE(CondHDF5ESSource);
