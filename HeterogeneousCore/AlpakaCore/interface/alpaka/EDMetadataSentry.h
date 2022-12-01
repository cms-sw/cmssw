#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadataSentry_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadataSentry_h

#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    /**
     * Helper class to be used in produce()
     *
     * TODO: not really a sentry as it doesn't do anything special in its destructor. Better name?
     */
    class EDMetadataSentry {
    public:
      // For normal module
      EDMetadataSentry(edm::StreamID stream);

      // For ExternalWork-module's produce()
      EDMetadataSentry(std::shared_ptr<EDMetadata> metadata) : metadata_(std::move(metadata)) {}

      EDMetadataSentry(EDMetadataSentry const&) = delete;
      EDMetadataSentry& operator=(EDMetadataSentry const&) = delete;
      EDMetadataSentry(EDMetadataSentry&&) = delete;
      EDMetadataSentry& operator=(EDMetadataSentry&&) = delete;

      std::shared_ptr<EDMetadata> metadata() { return metadata_; }

      void finish();

    private:
      std::shared_ptr<EDMetadata> metadata_;
    };
  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
