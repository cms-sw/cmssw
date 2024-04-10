#ifndef RecoLocalTracker_SiPixelRecHits_interface_PixelCPEFastParamsHost_h
#define RecoLocalTracker_SiPixelRecHits_interface_PixelCPEFastParamsHost_h

#include <alpaka/alpaka.hpp>

#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

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

  // non-copyable
  PixelCPEFastParamsHost(PixelCPEFastParamsHost const&) = delete;
  PixelCPEFastParamsHost& operator=(PixelCPEFastParamsHost const&) = delete;

  // movable
  PixelCPEFastParamsHost(PixelCPEFastParamsHost&&) = default;
  PixelCPEFastParamsHost& operator=(PixelCPEFastParamsHost&&) = default;

  // default destructor
  ~PixelCPEFastParamsHost() override = default;

  // access the buffer
  Buffer buffer() { return buffer_; }
  ConstBuffer buffer() const { return buffer_; }
  ConstBuffer const_buffer() const { return buffer_; }

  auto size() const { return alpaka::getExtentProduct(buffer_); }

  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* data() const { return buffer_.data(); }

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  LocalPoint localPosition(DetParam const& theDetParam, ClusterParam& theClusterParam) const override;
  LocalError localError(DetParam const& theDetParam, ClusterParam& theClusterParam) const override;

  void errorFromTemplates(DetParam const& theDetParam, ClusterParamGeneric& theClusterParam, float qclus) const;

  std::vector<SiPixelGenErrorStore> thePixelGenError_;

  void fillParamsForDevice();

  Buffer buffer_;
};

#endif  // RecoLocalTracker_SiPixelRecHits_interface_PixelCPEFastParamsHost_h
