#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

void toGlobalWrapper(SOAFrame<float> const *frame,
                     float const *xl,
                     float const *yl,
                     float *x,
                     float *y,
                     float *z,
                     float const *le,
                     float *ge,
                     uint32_t n);

int main(void) {
  cms::cudatest::requireDevices();

  typedef float T;
  typedef TkRotation<T> Rotation;
  typedef SOARotation<T> SRotation;
  typedef GloballyPositioned<T> Frame;
  typedef SOAFrame<T> SFrame;
  typedef typename Frame::PositionType Position;
  typedef typename Frame::GlobalVector GlobalVector;
  typedef typename Frame::GlobalPoint GlobalPoint;
  typedef typename Frame::LocalVector LocalVector;
  typedef typename Frame::LocalPoint LocalPoint;

  constexpr uint32_t size = 10000;
  constexpr uint32_t size32 = size * sizeof(float);

  float xl[size], yl[size];
  float x[size], y[size], z[size];

  // errors
  float le[3 * size];
  float ge[6 * size];

  auto d_xl = cms::cuda::make_device_unique<float[]>(size, nullptr);
  auto d_yl = cms::cuda::make_device_unique<float[]>(size, nullptr);

  auto d_x = cms::cuda::make_device_unique<float[]>(size, nullptr);
  auto d_y = cms::cuda::make_device_unique<float[]>(size, nullptr);
  auto d_z = cms::cuda::make_device_unique<float[]>(size, nullptr);

  auto d_le = cms::cuda::make_device_unique<float[]>(3 * size, nullptr);
  auto d_ge = cms::cuda::make_device_unique<float[]>(6 * size, nullptr);

  double a = 0.01;
  double ca = std::cos(a);
  double sa = std::sin(a);

  Rotation r1(ca, sa, 0, -sa, ca, 0, 0, 0, 1);
  Frame f1(Position(2, 3, 4), r1);
  std::cout << "f1.position() " << f1.position() << std::endl;
  std::cout << "f1.rotation() " << '\n' << f1.rotation() << std::endl;

  SFrame sf1(f1.position().x(), f1.position().y(), f1.position().z(), f1.rotation());

  auto d_sf = cms::cuda::make_device_unique<char[]>(sizeof(SFrame), nullptr);
  cudaCheck(cudaMemcpy(d_sf.get(), &sf1, sizeof(SFrame), cudaMemcpyHostToDevice));

  for (auto i = 0U; i < size; ++i) {
    xl[i] = yl[i] = 0.1f * float(i) - float(size / 2);
    le[3 * i] = 0.01f;
    le[3 * i + 2] = (i > size / 2) ? 1.f : 0.04f;
    le[2 * i + 1] = 0.;
  }
  std::random_shuffle(xl, xl + size);
  std::random_shuffle(yl, yl + size);

  cudaCheck(cudaMemcpy(d_xl.get(), xl, size32, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_yl.get(), yl, size32, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_le.get(), le, 3 * size32, cudaMemcpyHostToDevice));

  toGlobalWrapper((SFrame const *)(d_sf.get()),
                  d_xl.get(),
                  d_yl.get(),
                  d_x.get(),
                  d_y.get(),
                  d_z.get(),
                  d_le.get(),
                  d_ge.get(),
                  size);
  cudaCheck(cudaMemcpy(x, d_x.get(), size32, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(y, d_y.get(), size32, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(z, d_z.get(), size32, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(ge, d_ge.get(), 6 * size32, cudaMemcpyDeviceToHost));

  float eps = 0.;
  for (auto i = 0U; i < size; ++i) {
    auto gp = f1.toGlobal(LocalPoint(xl[i], yl[i]));
    eps = std::max(eps, std::abs(x[i] - gp.x()));
    eps = std::max(eps, std::abs(y[i] - gp.y()));
    eps = std::max(eps, std::abs(z[i] - gp.z()));
  }

  std::cout << "max eps " << eps << std::endl;

  return 0;
}
