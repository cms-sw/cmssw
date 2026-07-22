#ifndef DataFormats_VertexSoA_interface_alpaka_TrackForVertexDeviceCollection_h
#define DataFormats_VertexSoA_interface_alpaka_TrackForVertexDeviceCollection_h

#include "DataFormats/VertexSoA/interface/TrackForVertexHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/TrackForVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {
  using ::reco::TrackForVertexHostCollection;

  using TrackForVertexDeviceCollection = PortableCollection<::reco::TrackForVertexSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif  // DataFormats_VertexSoA_interface_alpaka_TrackForVertexDeviceCollection_h
