// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     HDF5DataProxy
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 13:52:59 GMT
//

// system include files
#include <iostream>
#include <H5Cpp.h>
#include <cassert>

// user include files
#include "HDF5DataProxy.h"
#include "convertSyncValue.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HDF5DataProxy::HDF5DataProxy(edm::SerialTaskQueue* iQueue,
                             std::mutex* iMutex,
                             std::unique_ptr<cond::serialization::SerializationHelperBase> iHelper,
                             H5::H5File* iFile,
                             cond::hdf5::Record const* iRecord,
                             cond::hdf5::DataProduct const* iDataProduct)
    : edm::eventsetup::ESSourceDataProxyNonConcurrentBase(iQueue, iMutex),
      helper_(std::move(iHelper)),
      file_(iFile),
      record_(iRecord),
      dataProduct_(iDataProduct) {}

// HDF5DataProxy::HDF5DataProxy(const HDF5DataProxy& rhs)
// {
//    // do actual copying here;
// }

HDF5DataProxy::~HDF5DataProxy() {}

//
// member functions
//
void HDF5DataProxy::prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails iRecord) {
  using namespace cond::hdf5;
  auto iov = iRecord.validityInterval();
  auto firstSync = convertSyncValue(iov.first(), record_->iovIsRunLumi_);

  auto itFound = findMatchingFirst(record_->iovFirsts_, firstSync);
  assert(itFound != record_->iovFirsts_.end());

  auto payloadRef = dataProduct_->payloadForIOVs_[itFound - record_->iovFirsts_.begin()];
  assert(file_->getRefObjType(&payloadRef) == H5O_TYPE_DATASET);
  H5::DataSet dataset(*file_, &payloadRef);
  std::vector<char> buffer;
  {
    const auto dataSpace = dataset.getSpace();
    assert(dataSpace.isSimple());
    assert(dataSpace.getSimpleExtentNdims() == 1);
    hsize_t size[1];
    dataSpace.getSimpleExtentDims(size);
    buffer.resize(size[0]);
  }
  if (buffer.empty()) {
    std::cout << "buffer empty for " << iKey.type().name() << std::endl;
    return;
  }

  dataset.read(&buffer[0], H5::PredType::STD_I8LE);

  std::stringbuf sBuffer;
  sBuffer.pubsetbuf(&buffer[0], buffer.size());
  std::string typeName;
  {
    auto const typeAttr = dataset.openAttribute("type");
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    typeAttr.read(str_type, typeName);
  }
  data_ = helper_->deserialize(sBuffer, typeName);
  if (data_.get() == nullptr) {
    std::cout << "failed to deserialize " << iKey.type().name() << " buffer size:" << buffer.size() << " type: '"
              << typeName << "'"
              << " in " << iRecord.key().name() << std::endl;
  } else {
    std::cout << "good " << iKey.type().name() << " '" << iKey.name().value() << "' firstSync:" << firstSync.high_
              << " " << firstSync.low_ << std::endl;
  }
}

void HDF5DataProxy::invalidateCache() {
  ESSourceDataProxyBase::invalidateCache();
  data_ = cond::serialization::unique_void_ptr();
}

//
// const member functions
//
void const* HDF5DataProxy::getAfterPrefetchImpl() const { return data_.get(); }

//
// static member functions
//
