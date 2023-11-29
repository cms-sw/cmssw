#ifndef DataFormats_PixelCPEFastParams_interface_PixelCPEFastParamsHost_h
#define DataFormats_PixelCPEFastParams_interface_PixelCPEFastParamsHost_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"

#include "pixelCPEforDevice.h"

namespace pixelCPEforDevice {

  constexpr float micronsToCm = 1.0e-4;

}

template <typename TrackerTraits>
class PixelCPEFastParamsHost : public PixelCPEGenericBase {
public:
  using Buffer = cms::alpakatools::host_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;

  PixelCPEFastParamsHost(edm::ParameterSet const& conf,
                         const MagneticField* mag,
                         const TrackerGeometry& geom,
                         const TrackerTopology& ttopo,
                         const SiPixelLorentzAngle* lorentzAngle,
                         const SiPixelGenErrorDBObject* genErrorDBObject,
                         const SiPixelLorentzAngle* lorentzAngleWidth);

  Buffer buffer() { return buffer_; }
  ConstBuffer buffer() const { return buffer_; }
  ConstBuffer const_buffer() const { return buffer_; }
  auto size() const { return alpaka::getExtentProduct(buffer_); }

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  LocalPoint localPosition(DetParam const& theDetParam, ClusterParam& theClusterParam) const override;
  LocalError localError(DetParam const& theDetParam, ClusterParam& theClusterParam) const override;

  void errorFromTemplates(DetParam const& theDetParam, ClusterParamGeneric& theClusterParam, float qclus) const;

  std::vector<SiPixelGenErrorStore> thePixelGenError_;

  void fillParamsForDevice();

  Buffer buffer_;
};
// }  // namespace pixelCPEforDevice

#endif
