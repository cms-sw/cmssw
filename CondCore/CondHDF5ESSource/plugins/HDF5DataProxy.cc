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
#include <fstream>
#include <cassert>
#include "zlib.h"

// user include files
#include "HDF5DataProxy.h"
#include "convertSyncValue.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
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
                             std::unique_ptr<cond::serialization::SerializationHelperBase> iHelper,
                             cms::h5::File const* iFile,
                             std::string const& iFileName,
                             cond::hdf5::Record const* iRecord,
                             cond::hdf5::DataProduct const* iDataProduct)
    : edm::eventsetup::ESSourceDataProxyBase(),
      queue_(iQueue),
      helper_(std::move(iHelper)),
      file_(iFile),
      fileName_(iFileName),
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

void HDF5DataProxy::prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                                      edm::eventsetup::EventSetupRecordImpl const& iRecord,
                                      edm::eventsetup::DataKey const& iKey,
                                      edm::EventSetupImpl const*,
                                      edm::ServiceToken const&,
                                      edm::ESParentContext const& iParent) {
  prefetchAsyncImplTemplate(
      [this, iov = iRecord.validityInterval()](auto& iGroup, auto iActivity) {
        queue_->push(iGroup, [this, &iGroup, act = std::move(iActivity), iov] {
          try {
            auto index = indexForInterval(iov);

            readFromHDF5api(index);
            iGroup.run(std::move(act));
            exceptPtr_ = {};
          } catch (...) {
            exceptPtr_ = std::current_exception();
          }
        });
      },
      []() { return true; },
      std::move(iTask),
      iRecord,
      iKey,
      iParent);
}

std::ptrdiff_t HDF5DataProxy::indexForInterval(edm::ValidityInterval const& iIOV) const {
  using namespace cond::hdf5;
  auto firstSync = convertSyncValue(iIOV.first(), record_->iovIsRunLumi_);

  auto itFound = findMatchingFirst(record_->iovFirsts_, firstSync);
  assert(itFound != record_->iovFirsts_.end());

  return itFound - record_->iovFirsts_.begin();
}

void HDF5DataProxy::readFromHDF5api(std::ptrdiff_t iIndex) {
  auto payloadRef = dataProduct_->payloadForIOVs_[iIndex];
  auto ds = file_->derefDataSet(payloadRef);
  storageSize_ = ds->storageSize();
  if (storageSize_ == 0) {
    return;
  }

  fileOffset_ = ds->fileOffset();
  memSize_ = ds->findAttribute("memsize")->readUInt32();
  type_ = ds->findAttribute("type")->readString();
}

void HDF5DataProxy::prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails iRecord) {
  if (exceptPtr_) {
    rethrow_exception(exceptPtr_);
  }
  if (storageSize_ == 0) {
    return;
  }
  threadFriendlyPrefetch(fileOffset_, storageSize_, memSize_, type_);
}

std::vector<char> HDF5DataProxy::decompress(std::vector<char> compressedBuffer, std::size_t iMemSize) const {
  std::vector<char> buffer;
  if (iMemSize == compressedBuffer.size()) {
    //memory was not compressed
    //std::cout <<"NOT COMPRESSED"<<std::endl;
    buffer = std::move(compressedBuffer);
  } else {
    //zlib compression was used
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    auto ret = inflateInit(&strm);
    assert(ret == Z_OK);

    strm.avail_in = compressedBuffer.size();
    strm.next_in = reinterpret_cast<unsigned char*>(compressedBuffer.data());

    buffer = std::vector<char>(iMemSize);
    strm.avail_out = buffer.size();
    strm.next_out = reinterpret_cast<unsigned char*>(buffer.data());
    ret = inflate(&strm, Z_FINISH);
    assert(ret != Z_STREAM_ERROR);
    //if(ret != Z_STREAM_END) {std::cout <<"mem "<<memSize<<" "<<ret<<" out "<<strm.avail_out<<std::endl;}
    assert(ret == Z_STREAM_END);

    (void)inflateEnd(&strm);
  }
  return buffer;
}

void HDF5DataProxy::threadFriendlyPrefetch(uint64_t iFileOffset,
                                           std::size_t iStorageSize,
                                           std::size_t iMemSize,
                                           const std::string& iTypeName) {
  //Done interacting with the hdf5 API

  //std::cout <<" prefetch "<<dataProduct_->fileOffsets_[index]<<" "<<dataProduct_->storageSizes_[index]<<" "<<memSize<<std::endl;
  std::vector<char> compressedBuffer(iStorageSize);
  std::fstream file(fileName_.c_str());
  file.seekg(iFileOffset);
  file.read(compressedBuffer.data(), compressedBuffer.size());

  std::vector<char> buffer = decompress(std::move(compressedBuffer), iMemSize);

  std::stringbuf sBuffer;
  sBuffer.pubsetbuf(&buffer[0], buffer.size());
  data_ = helper_->deserialize(sBuffer, iTypeName);
  if (data_.get() == nullptr) {
    throw cms::Exception("H5CondFailedDeserialization")
        << "failed to deserialize: buffer size:" << buffer.size() << " type: '" << iTypeName << "'";
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
