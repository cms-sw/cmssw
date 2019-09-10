#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAView* hits);

}

int main() {
  exitSansCUDADevices();

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  auto nHits = 200;
  TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, stream);

  testTrackingRecHit2D::runKernels(tkhit.view());

  return 0;
}
