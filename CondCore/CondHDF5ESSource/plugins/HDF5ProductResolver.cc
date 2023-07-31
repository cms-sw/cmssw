// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     HDF5ProductResolver
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
#include "lzma.h"

// user include files
#include "HDF5ProductResolver.h"
#include "convertSyncValue.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
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
HDF5ProductResolver::HDF5ProductResolver(edm::SerialTaskQueue* iQueue,
                                         std::unique_ptr<cond::serialization::SerializationHelperBase> iHelper,
                                         cms::h5::File const* iFile,
                                         std::string const& iFileName,
                                         cond::hdf5::Compression iCompression,
                                         cond::hdf5::Record const* iRecord,
                                         cond::hdf5::DataProduct const* iDataProduct)
    : edm::eventsetup::ESSourceProductResolverBase(),
      queue_(iQueue),
      helper_(std::move(iHelper)),
      file_(iFile),
      fileName_(iFileName),
      record_(iRecord),
      dataProduct_(iDataProduct),
      compression_(iCompression) {}

// HDF5ProductResolver::HDF5ProductResolver(const HDF5ProductResolver& rhs)
// {
//    // do actual copying here;
// }

HDF5ProductResolver::~HDF5ProductResolver() {}

//
// member functions
//

void HDF5ProductResolver::prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                                            edm::eventsetup::EventSetupRecordImpl const& iRecord,
                                            edm::eventsetup::DataKey const& iKey,
                                            edm::EventSetupImpl const*,
                                            edm::ServiceToken const&,
                                            edm::ESParentContext const& iParent) {
  prefetchAsyncImplTemplate(
      [this, iov = iRecord.validityInterval(), iParent, &iRecord](auto& iGroup, auto iActivity) {
        queue_->push(iGroup, [this, &iGroup, act = std::move(iActivity), iov, iParent, &iRecord] {
          try {
            edm::ESModuleCallingContext context(
                providerDescription(), edm::ESModuleCallingContext::State::kRunning, iParent);
            iRecord.activityRegistry()->preESModuleSignal_.emit(iRecord.key(), context);
            struct EndGuard {
              EndGuard(edm::eventsetup::EventSetupRecordImpl const& iRecord,
                       edm::ESModuleCallingContext const& iContext)
                  : record_{iRecord}, context_{iContext} {}
              ~EndGuard() { record_.activityRegistry()->postESModuleSignal_.emit(record_.key(), context_); }
              edm::eventsetup::EventSetupRecordImpl const& record_;
              edm::ESModuleCallingContext const& context_;
            } guardAR(iRecord, context);

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

std::ptrdiff_t HDF5ProductResolver::indexForInterval(edm::ValidityInterval const& iIOV) const {
  using namespace cond::hdf5;
  auto firstSync = convertSyncValue(iIOV.first(), record_->iovIsRunLumi_);

  auto itFound = findMatchingFirst(record_->iovFirsts_, firstSync);
  assert(itFound != record_->iovFirsts_.end());

  return itFound - record_->iovFirsts_.begin();
}

void HDF5ProductResolver::readFromHDF5api(std::ptrdiff_t iIndex) {
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

void HDF5ProductResolver::prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails iRecord) {
  if (exceptPtr_) {
    rethrow_exception(exceptPtr_);
  }
  if (storageSize_ == 0) {
    return;
  }
  threadFriendlyPrefetch(fileOffset_, storageSize_, memSize_, type_);
}

std::vector<char> HDF5ProductResolver::decompress_zlib(std::vector<char> compressedBuffer, std::size_t iMemSize) const {
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

std::vector<char> HDF5ProductResolver::decompress_lzma(std::vector<char> compressedBuffer, std::size_t iMemSize) const {
  std::vector<char> buffer;
  if (iMemSize == compressedBuffer.size()) {
    //memory was not compressed
    //std::cout <<"NOT COMPRESSED"<<std::endl;
    buffer = std::move(compressedBuffer);
  } else {
    // code 'cribbed' from ROOT
    lzma_stream stream = LZMA_STREAM_INIT;

    auto returnStatus = lzma_stream_decoder(&stream, UINT64_MAX, 0U);
    if (returnStatus != LZMA_OK) {
      throw cms::Exception("H5CondFailedDecompress") << "failed to setup lzma";
    }

    stream.next_in = reinterpret_cast<uint8_t*>(compressedBuffer.data());
    stream.avail_in = compressedBuffer.size();

    buffer = std::vector<char>(iMemSize);
    stream.next_out = reinterpret_cast<uint8_t*>(buffer.data());
    stream.avail_out = buffer.size();

    returnStatus = lzma_code(&stream, LZMA_FINISH);
    lzma_end(&stream);
    if (returnStatus != LZMA_STREAM_END) {
      throw cms::Exception("H5CondFailedDecompress") << "failed to decompress buffer using lzma";
    }
    lzma_end(&stream);
  }
  return buffer;
}

void HDF5ProductResolver::threadFriendlyPrefetch(uint64_t iFileOffset,
                                                 std::size_t iStorageSize,
                                                 std::size_t iMemSize,
                                                 const std::string& iTypeName) {
  //Done interacting with the hdf5 API

  //std::cout <<" prefetch "<<dataProduct_->fileOffsets_[index]<<" "<<dataProduct_->storageSizes_[index]<<" "<<memSize<<std::endl;
  std::vector<char> compressedBuffer(iStorageSize);
  std::fstream file(fileName_.c_str());
  file.seekg(iFileOffset);
  file.read(compressedBuffer.data(), compressedBuffer.size());

  std::vector<char> buffer;
  if (compression_ == cond::hdf5::Compression::kZLIB) {
    buffer = decompress_zlib(std::move(compressedBuffer), iMemSize);
  } else if (compression_ == cond::hdf5::Compression::kLZMA) {
    buffer = decompress_lzma(std::move(compressedBuffer), iMemSize);
  } else {
    buffer = std::move(compressedBuffer);
  }

  std::stringbuf sBuffer;
  sBuffer.pubsetbuf(&buffer[0], buffer.size());
  data_ = helper_->deserialize(sBuffer, iTypeName);
  if (data_.get() == nullptr) {
    throw cms::Exception("H5CondFailedDeserialization")
        << "failed to deserialize: buffer size:" << buffer.size() << " type: '" << iTypeName << "'";
  }
}

void HDF5ProductResolver::invalidateCache() {
  ESSourceProductResolverBase::invalidateCache();
  data_ = cond::serialization::unique_void_ptr();
}

//
// const member functions
//
void const* HDF5ProductResolver::getAfterPrefetchImpl() const { return data_.get(); }

//
// static member functions
//
