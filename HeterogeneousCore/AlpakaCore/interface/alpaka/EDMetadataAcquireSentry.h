#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadataAcquireSentry_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadataAcquireSentry_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    /**
     * Helper class to be used in acquire()
     *
     * TODO: not really a sentry as it doesn't do anything special in its destructor. Better name?
     */
    class EDMetadataAcquireSentry {
    public:
      // TODO: WaitingTaskWithArenaHolder not really needed for host synchronous case
      // Constructor overload to be called from acquire()
      EDMetadataAcquireSentry(edm::StreamID stream, edm::WaitingTaskWithArenaHolder holder);

      // Constructor overload to be called from registerTransformAsync()
      EDMetadataAcquireSentry(Device const& device, edm::WaitingTaskWithArenaHolder holder);

      EDMetadataAcquireSentry(EDMetadataAcquireSentry const&) = delete;
      EDMetadataAcquireSentry& operator=(EDMetadataAcquireSentry const&) = delete;
      EDMetadataAcquireSentry(EDMetadataAcquireSentry&&) = delete;
      EDMetadataAcquireSentry& operator=(EDMetadataAcquireSentry&&) = delete;

      std::shared_ptr<EDMetadata> metadata() { return metadata_; }

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      // all synchronous backends
      std::shared_ptr<EDMetadata> finish() { return std::move(metadata_); }
#else
      // all asynchronous backends
      std::shared_ptr<EDMetadata> finish();
#endif

    private:
      std::shared_ptr<EDMetadata> metadata_;

      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
    };
  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
