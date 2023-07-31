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
                             cms::h5::File const* iFile,
                             std::string const& iFileName,
                             cond::hdf5::Record const* iRecord,
                             cond::hdf5::DataProduct const* iDataProduct)
    : edm::eventsetup::ESSourceDataProxyNonConcurrentBase(iQueue, iMutex),
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
void HDF5DataProxy::prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails iRecord) {
  using namespace cond::hdf5;
  auto iov = iRecord.validityInterval();
  auto firstSync = convertSyncValue(iov.first(), record_->iovIsRunLumi_);

  auto itFound = findMatchingFirst(record_->iovFirsts_, firstSync);
  assert(itFound != record_->iovFirsts_.end());

  auto index = itFound - record_->iovFirsts_.begin();

  auto payloadRef = dataProduct_->payloadForIOVs_[index];

  auto ds = file_->derefDataSet(payloadRef);
  auto storageSize = ds->storageSize();
  if (storageSize == 0) {
    return;
  }

  auto fileOffset = ds->fileOffset();
  auto memSize = ds->findAttribute("memsize")->readUInt32();
  auto typeName = ds->findAttribute("type")->readString();

  //Done interacting with the hdf5 API

  //std::cout <<" prefetch "<<dataProduct_->fileOffsets_[index]<<" "<<dataProduct_->storageSizes_[index]<<" "<<memSize<<std::endl;
  std::vector<char> compressedBuffer(storageSize);
  std::fstream file(fileName_.c_str());
  file.seekg(fileOffset);
  file.read(compressedBuffer.data(), compressedBuffer.size());

  std::vector<char> buffer;
  if (memSize == compressedBuffer.size()) {
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

    buffer = std::vector<char>(memSize);
    strm.avail_out = buffer.size();
    strm.next_out = reinterpret_cast<unsigned char*>(buffer.data());
    ret = inflate(&strm, Z_FINISH);
    assert(ret != Z_STREAM_ERROR);
    //if(ret != Z_STREAM_END) {std::cout <<"mem "<<memSize<<" "<<ret<<" out "<<strm.avail_out<<std::endl;}
    assert(ret == Z_STREAM_END);

    (void)inflateEnd(&strm);
  }

  std::stringbuf sBuffer;
  sBuffer.pubsetbuf(&buffer[0], buffer.size());
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
