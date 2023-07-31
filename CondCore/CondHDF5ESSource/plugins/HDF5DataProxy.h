#ifndef CondCore_HDF5ESSource_HDF5DataProxy_h
#define CondCore_HDF5ESSource_HDF5DataProxy_h
// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     HDF5DataProxy
//
/**\class HDF5DataProxy HDF5DataProxy.h "HDF5DataProxy.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 13:52:57 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESSourceDataProxyBase.h"
#include "CondFormats/SerializationHelper/interface/SerializationHelperBase.h"
#include "Record.h"
#include "DataProduct.h"
#include "h5_File.h"

// forward declarations
namespace edm {
  class SerialTaskQueue;
}

class HDF5DataProxy : public edm::eventsetup::ESSourceDataProxyBase {
public:
  HDF5DataProxy(edm::SerialTaskQueue* iQueue,
                std::unique_ptr<cond::serialization::SerializationHelperBase>,
                cms::h5::File const* iFile,
                std::string const& iFileName,
                cond::hdf5::Record const* iRecord,
                cond::hdf5::DataProduct const* iDataProduct);
  ~HDF5DataProxy() final;

  HDF5DataProxy(const HDF5DataProxy&) = delete;                   // stop default
  const HDF5DataProxy& operator=(const HDF5DataProxy&) = delete;  // stop default

private:
private:
  void prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                         edm::eventsetup::EventSetupRecordImpl const& iES,
                         edm::eventsetup::DataKey const& iKey,
                         edm::EventSetupImpl const*,
                         edm::ServiceToken const&,
                         edm::ESParentContext const&) final;

  void invalidateCache() final;
  void prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails) final;
  void const* getAfterPrefetchImpl() const final;

  std::ptrdiff_t indexForInterval(edm::ValidityInterval const& iIOV) const;

  void readFromHDF5api(std::ptrdiff_t iIndex);
  void threadFriendlyPrefetch(uint64_t iFileOffset,
                              std::size_t iStorageSize,
                              std::size_t iMemSize,
                              const std::string& iType);

  std::vector<char> decompress(std::vector<char>, std::size_t iMemSize) const;
  // ---------- member data --------------------------------
  edm::SerialTaskQueue* queue_;
  cond::serialization::unique_void_ptr data_;
  std::unique_ptr<cond::serialization::SerializationHelperBase> helper_;
  cms::h5::File const* file_;
  std::string fileName_;
  cond::hdf5::Record const* record_;
  cond::hdf5::DataProduct const* dataProduct_;

  //temporaries
  uint64_t fileOffset_;
  std::string type_;
  std::size_t storageSize_;
  std::size_t memSize_;
  std::exception_ptr exceptPtr_;
};

#endif
