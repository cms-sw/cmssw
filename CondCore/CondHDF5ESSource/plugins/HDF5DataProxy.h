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
#include "FWCore/Framework/interface/ESSourceDataProxyNonConcurrentBase.h"
#include "CondFormats/SerializationHelper/interface/SerializationHelperBase.h"
#include "Record.h"
#include "DataProduct.h"
#include "h5_File.h"

// forward declarations

class HDF5DataProxy : public edm::eventsetup::ESSourceDataProxyNonConcurrentBase {
public:
  HDF5DataProxy(edm::SerialTaskQueue* iQueue,
                std::mutex* iMutex,
                std::unique_ptr<cond::serialization::SerializationHelperBase>,
                cms::h5::File* iFile,
                cond::hdf5::Record const* iRecord,
                cond::hdf5::DataProduct const* iDataProduct);
  ~HDF5DataProxy() final;

  HDF5DataProxy(const HDF5DataProxy&) = delete;                   // stop default
  const HDF5DataProxy& operator=(const HDF5DataProxy&) = delete;  // stop default

private:
  void invalidateCache() final;
  void prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails) final;
  void const* getAfterPrefetchImpl() const final;

  // ---------- member data --------------------------------
  cond::serialization::unique_void_ptr data_;
  std::unique_ptr<cond::serialization::SerializationHelperBase> helper_;
  cms::h5::File* file_;
  cond::hdf5::Record const* record_;
  cond::hdf5::DataProduct const* dataProduct_;
};

#endif
