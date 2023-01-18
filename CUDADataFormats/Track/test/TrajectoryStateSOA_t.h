#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using helper = TracksUtilities<pixelTopology::Phase1>;

__host__ __device__ Matrix5d loadCov(Vector5d const& e) {
  Matrix5d cov;
  for (int i = 0; i < 5; ++i)
    cov(i, i) = e(i) * e(i);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3 * std::sqrt(cov(i, i) * cov(j, j));  // this makes the matrix pos defined
      cov(i, j) = (i + j) % 2 ? -0.4 * v : 0.1 * v;
      cov(j, i) = cov(i, j);
    }
  }
  return cov;
}

template <typename TrackerTraits>
__global__ void testTSSoA(TrackSoAView<TrackerTraits> ts) {
  Vector5d par0;
  par0 << 0.2, 0.1, 3.5, 0.8, 0.1;
  Vector5d e0;
  e0 << 0.01, 0.01, 0.035, -0.03, -0.01;
  auto cov0 = loadCov(e0);

  int first = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = first; i < ts.metadata().size(); i += blockDim.x * gridDim.x) {
    helper::copyFromDense(ts, par0, cov0, i);
    Vector5d par1;
    Matrix5d cov1;
    helper::copyToDense(ts, par1, cov1, i);
    Vector5d delV = par1 - par0;
    Matrix5d delM = cov1 - cov0;
    for (int j = 0; j < 5; ++j) {
      assert(std::abs(delV(j)) < 1.e-5);
      for (auto k = j; k < 5; ++k) {
        assert(cov0(k, j) == cov0(j, k));
        assert(cov1(k, j) == cov1(j, k));
        assert(std::abs(delM(k, j)) < 1.e-5);
      }
    }
  }
}

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#endif

int main() {
#ifdef __CUDACC__
  cms::cudatest::requireDevices();
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif

#ifdef __CUDACC__
  // Since we are going to copy data from ts_d to ts_h, we
  // need to initialize the Host collection with a stream.
  TrackSoAHeterogeneousHost<pixelTopology::Phase1> ts_h(stream);
  TrackSoAHeterogeneousDevice<pixelTopology::Phase1> ts_d(stream);
#else
  // If CUDA is not available, Host collection must not be initialized
  // with a stream.
  TrackSoAHeterogeneousHost<pixelTopology::Phase1> ts_h;
#endif

#ifdef __CUDACC__
  testTSSoA<pixelTopology::Phase1><<<1, 64, 0, stream>>>(ts_d.view());
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpyAsync(
      ts_h.buffer().get(), ts_d.const_buffer().get(), ts_d.bufferSize(), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaStreamSynchronize(stream));
#else
  testTSSoA<pixelTopology::Phase1>(ts_h.view());
#endif
}
