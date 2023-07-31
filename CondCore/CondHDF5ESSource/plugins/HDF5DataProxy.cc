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
#include <cassert>

// user include files
#include "HDF5DataProxy.h"
#include "convertSyncValue.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "h5_DataSet.h"
#include "h5_Attribute.h"

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
                             cms::h5::File* iFile,
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
  auto dataset = file_->derefDataSet(payloadRef);
  std::vector<char> buffer = dataset->readBytes();
  if (buffer.empty()) {
    return;
  }

  std::stringbuf sBuffer;
  sBuffer.pubsetbuf(&buffer[0], buffer.size());
  std::string typeName;
  {
    auto const typeAttr = dataset->findAttribute("type");
    typeName = typeAttr->readString();
  }
  data_ = helper_->deserialize(sBuffer, typeName);
  if (data_.get() == nullptr) {
    throw cms::Exception("H5CondFailedDeserialization")
        << "failed to deserialize " << iKey.type().name() << " buffer size:" << buffer.size() << " type: '" << typeName
        << "'"
        << " in " << iRecord.key().name() << std::endl;
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
