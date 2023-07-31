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
HDF5DataProxy::HDF5DataProxy(std::unique_ptr<cond::serialization::SerializationHelperBase> iHelper,
                             std::string iFileName,
                             cond::hdf5::Record const* iRecord,
                             cond::hdf5::DataProduct const* iDataProduct)
    : helper_(std::move(iHelper)), filename_(std::move(iFileName)), record_(iRecord), dataProduct_(iDataProduct) {}

// HDF5DataProxy::HDF5DataProxy(const HDF5DataProxy& rhs)
// {
//    // do actual copying here;
// }

HDF5DataProxy::~HDF5DataProxy() {}

//
// member functions
//
void HDF5DataProxy::prefetchAsyncImpl(edm::WaitingTaskHolder,
                                      edm::eventsetup::EventSetupRecordImpl const& iRecord,
                                      edm::eventsetup::DataKey const& iKey,
                                      edm::EventSetupImpl const*,
                                      edm::ServiceToken const&,
                                      edm::ESParentContext const&) {
  using namespace cond::hdf5;
  auto iov = iRecord.validityInterval();
  auto firstSync = convertSyncValue(iov.first(), record_->iovIsRunLumi_);

  auto itFound = findMatchingFirst(record_->iovFirsts_, firstSync);

  H5::H5File file(filename_, H5F_ACC_RDONLY);
  auto payloadRef = dataProduct_->payloadForIOVs_[itFound - record_->iovFirsts_.begin()];
  assert(file.getRefObjType(&payloadRef) == H5O_TYPE_DATASET);
  H5::DataSet dataset(file, &payloadRef);
  std::vector<char> buffer;
  {
    const auto dataSpace = dataset.getSpace();
    assert(dataSpace.isSimple());
    assert(dataSpace.getSimpleExtentNdims() == 1);
    hsize_t size[1];
    dataSpace.getSimpleExtentDims(size);
    buffer.resize(size[0]);
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
}

void HDF5DataProxy::invalidateCache() { data_ = cond::serialization::unique_void_ptr(); }

//
// const member functions
//
void const* HDF5DataProxy::getAfterPrefetchImpl() const { return data_.get(); }

//
// static member functions
//
