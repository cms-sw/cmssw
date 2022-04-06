#include <Eigen/Core>
#include "CUDADataFormats/Common/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/SoAView.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include <memory>
#include <cstdlib>
#include <Eigen/Dense>

// Test SoA stores and view.
// Use cases
// Multiple stores in a buffer
// Scalars, Columns of scalars and of Eigen vectors
// View to each of them, from one and multiple stores.

GENERATE_SOA_LAYOUT_AND_VIEW(SoAHostDeviceLayoutTemplate,
                             SoAHostDeviceViewTemplate,
                             // predefined static scalars
                             // size_t size;
                             // size_t alignment;

                             // columns: one value per element
                             SOA_COLUMN(double, x),
                             SOA_COLUMN(double, y),
                             SOA_COLUMN(double, z),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
                             // scalars: one value for the whole structure
                             SOA_SCALAR(const char*, description),
                             SOA_SCALAR(uint32_t, someNumber))

using SoAHostDeviceLayout = SoAHostDeviceLayoutTemplate<>;
using SoAHostDeviceView =
    SoAHostDeviceViewTemplate<cms::soa::CacheLineSize::NvidiaGPU, cms::soa::AlignmentEnforcement::Enforced>;

GENERATE_SOA_LAYOUT_AND_VIEW(SoADeviceOnlyLayoutTemplate,
                             SoADeviceOnlyViewTemplate,
                             SOA_COLUMN(uint16_t, color),
                             SOA_COLUMN(double, value),
                             SOA_COLUMN(double*, py),
                             SOA_COLUMN(uint32_t, count),
                             SOA_COLUMN(uint32_t, anotherCount))

using SoADeviceOnlyLayout = SoADeviceOnlyLayoutTemplate<>;
using SoADeviceOnlyView =
    SoADeviceOnlyViewTemplate<cms::soa::CacheLineSize::NvidiaGPU, cms::soa::AlignmentEnforcement::Enforced>;

// A 1 to 1 view of the store (except for unsupported types).
GENERATE_SOA_VIEW(SoAFullDeviceViewTemplate,
                  SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(SoAHostDeviceLayout, soaHD),
                                       SOA_VIEW_LAYOUT(SoADeviceOnlyLayout, soaDO)),
                  SOA_VIEW_LAYOUT_LIST(SOA_VIEW_VALUE(soaHD, x),
                                       SOA_VIEW_VALUE(soaHD, y),
                                       SOA_VIEW_VALUE(soaHD, z),
                                       SOA_VIEW_VALUE(soaDO, color),
                                       SOA_VIEW_VALUE(soaDO, value),
                                       SOA_VIEW_VALUE(soaDO, py),
                                       SOA_VIEW_VALUE(soaDO, count),
                                       SOA_VIEW_VALUE(soaDO, anotherCount),
                                       SOA_VIEW_VALUE(soaHD, description),
                                       SOA_VIEW_VALUE(soaHD, someNumber)))

using SoAFullDeviceView =
    SoAFullDeviceViewTemplate<cms::soa::CacheLineSize::NvidiaGPU, cms::soa::AlignmentEnforcement::Enforced>;

// Eigen cross product kernel (on store)
__global__ void crossProduct(SoAHostDeviceView soa, const unsigned int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=numElements) return;
  auto si = soa[i];
  si.r() = si.a().cross(si.b());
}

// Device-only producer kernel
__global__ void producerKernel(SoAFullDeviceView soa, const unsigned int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=numElements) return;
  auto si = soa[i];
  si.color() &= 0x55 << i % (sizeof(si.color()) - sizeof(char));
  si.value() = sqrt(si.x() * si.x() + si.y() * si.y() + si.z() * si.z());
}

// Device-only consumer with result in host-device area
__global__ void consumerKernel(SoAFullDeviceView soa, const unsigned int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=numElements) return;
  auto si = soa[i];
  si.x() = si.color() * si.value();
}

// Get a view like the default, except for range checking
using RangeCheckingHostDeviceView = SoAHostDeviceViewTemplate<SoAHostDeviceView::byteAlignment,
                                                              SoAHostDeviceView::alignmentEnforcement,
                                                              SoAHostDeviceView::restrictQualify,
                                                              cms::soa::RangeChecking::Enabled>;

// We expect to just run one thread.
__global__ void rangeCheckKernel(RangeCheckingHostDeviceView soa) {
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  printf("About to fail range check in CUDA thread: %d\n", threadIdx.x);
#endif
  [[maybe_unused]] auto si = soa[soa.soaMetadata().size()];
  printf("We should not have reached here\n");
}

int main(void) {
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Non-aligned number of elements to check alignment features.
  constexpr unsigned int numElements = 65537;

  // Allocate buffer and store on host
  size_t hostDeviceSize = SoAHostDeviceLayout::computeDataSize(numElements);
  std::byte * h_buf = nullptr;
  cudaCheck(cudaMallocHost(&h_buf, hostDeviceSize));
  SoAHostDeviceLayout h_soahdLayout(h_buf, numElements);
  SoAHostDeviceView h_soahd(h_soahdLayout);

  // Alocate buffer, stores and views on the device (single, shared buffer).
  size_t deviceOnlySize = SoADeviceOnlyLayout::computeDataSize(numElements);
  std::byte * d_buf = nullptr;
  cudaCheck(cudaMallocHost(&d_buf, hostDeviceSize + deviceOnlySize));
  SoAHostDeviceLayout d_soahdLayout(d_buf, numElements);
  SoADeviceOnlyLayout d_soadoLayout(d_soahdLayout.soaMetadata().nextByte(), numElements);
  SoAHostDeviceView d_soahdView(d_soahdLayout);
  SoAFullDeviceView d_soaFullView(d_soahdLayout, d_soadoLayout);

  // Assert column alignments
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_x()) % decltype(h_soahd)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_y()) % decltype(h_soahd)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_z()) % decltype(h_soahd)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_a()) % decltype(h_soahd)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_b()) % decltype(h_soahd)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_r()) % decltype(h_soahd)::byteAlignment);
  assert(0 ==
         reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_description()) % decltype(h_soahd)::byteAlignment);
  assert(0 ==
         reinterpret_cast<uintptr_t>(h_soahd.soaMetadata().addressOf_someNumber()) % decltype(h_soahd)::byteAlignment);

  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_x()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_y()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_z()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_a()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_b()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_r()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_description()) %
                  decltype(d_soahdLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soahdLayout.soaMetadata().addressOf_someNumber()) %
                  decltype(d_soahdLayout)::byteAlignment);

  assert(0 == reinterpret_cast<uintptr_t>(d_soadoLayout.soaMetadata().addressOf_color()) %
                  decltype(d_soadoLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soadoLayout.soaMetadata().addressOf_value()) %
                  decltype(d_soadoLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soadoLayout.soaMetadata().addressOf_py()) %
                  decltype(d_soadoLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soadoLayout.soaMetadata().addressOf_count()) %
                  decltype(d_soadoLayout)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soadoLayout.soaMetadata().addressOf_anotherCount()) %
                  decltype(d_soadoLayout)::byteAlignment);

  // Views should get the same alignment as the stores they refer to
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_x()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_y()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_z()) %
                  decltype(d_soaFullView)::byteAlignment);
  // Limitation of views: we have to get scalar member addresses via metadata.
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_description()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_someNumber()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_color()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_value()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_py()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_count()) %
                  decltype(d_soaFullView)::byteAlignment);
  assert(0 == reinterpret_cast<uintptr_t>(d_soaFullView.soaMetadata().addressOf_anotherCount()) %
                  decltype(d_soaFullView)::byteAlignment);

  // Initialize and fill the host buffer
  std::memset(h_soahdLayout.soaMetadata().data(), 0, hostDeviceSize);
  for (size_t i = 0; i < numElements; ++i) {
    auto si = h_soahd[i];
    si.x() = si.a()(0) = si.b()(2) = 1.0 * i + 1.0;
    si.y() = si.a()(1) = si.b()(1) = 2.0 * i;
    si.z() = si.a()(2) = si.b()(0) = 3.0 * i - 1.0;
  }
  auto& sn = h_soahd.someNumber();
  sn = numElements + 2;

  // Push to device
  cudaCheck(cudaMemcpyAsync(d_buf, h_buf, hostDeviceSize, cudaMemcpyDefault, stream));

  // Process on device
  crossProduct<<<(numElements + 255) / 256, 256, 0, stream>>>(d_soahdView, numElements);

  // Paint the device only with 0xFF initially
  cudaCheck(cudaMemsetAsync(d_soadoLayout.soaMetadata().data(), 0xFF, d_soadoLayout.soaMetadata().byteSize(), stream));
  
  // Produce to the device only area
  producerKernel<<<(numElements + 255) / 256, 256, 0, stream>>>(d_soaFullView, numElements);

  // Consume the device only area and generate a result on the host-device area
  consumerKernel<<<(numElements + 255) / 256, 256, 0, stream>>>(d_soaFullView, numElements);

  // Get result back
  cudaCheck(cudaMemcpyAsync(h_buf, d_buf, hostDeviceSize, cudaMemcpyDefault, stream));

  // Wait and validate.
  cudaCheck(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < numElements; ++i) {
    auto si = h_soahd[i];
    assert(si.r() == si.a().cross(si.b()));
    double initialX = 1.0 * i + 1.0;
    double initialY = 2.0 * i;
    double initialZ = 3.0 * i - 1.0;
    uint16_t expectedColor = 0x55 << i % (sizeof(uint16_t) - sizeof(char));
    double expectedX = expectedColor * sqrt(initialX * initialX + initialY * initialY + initialZ * initialZ);
    if (abs(si.x() - expectedX) / expectedX >= 2 * std::numeric_limits<double>::epsilon()) {
      std::cout << "X failed: for i=" << i << std::endl
                << "initialX=" << initialX << " initialY=" << initialY << " initialZ=" << initialZ << std::endl
                << "expectedX=" << expectedX << std::endl
                << "resultX=" << si.x() << " resultY=" << si.y() << " resultZ=" << si.z() << std::endl
                << "relativeDiff=" << abs(si.x() - expectedX) / expectedX
                << " epsilon=" << std::numeric_limits<double>::epsilon() << std::endl;
      assert(false);
    }
  }

  // Validation of range checking
  try {
    // Get a view like the default, except for range checking
    SoAHostDeviceViewTemplate<SoAHostDeviceView::byteAlignment,
                              SoAHostDeviceView::alignmentEnforcement,
                              SoAHostDeviceView::restrictQualify,
                              cms::soa::RangeChecking::Enabled>
        soa1viewRangeChecking(h_soahdLayout);
    // This should throw an exception
    [[maybe_unused]] auto si = soa1viewRangeChecking[soa1viewRangeChecking.soaMetadata().size()];
    assert(false);
  } catch (const std::out_of_range&) {
  }

  // Validation of range checking in a kernel
  // Get a view like the default, except for range checking
  RangeCheckingHostDeviceView soa1viewRangeChecking(d_soahdLayout);
  // This should throw an exception in the kernel
  try {
    rangeCheckKernel<<<1,1,0,stream>>>(soa1viewRangeChecking);
  } catch (const std::out_of_range&) {
    std::cout << "Exception received in enqueue." << std::endl;
  }

  // Wait and validate (that we failed).
  try {
    cudaCheck(cudaStreamSynchronize(stream));
  } catch (const std::runtime_error&) {
    std::cout << "Exception received in wait." << std::endl;
  }

  std::cout << "OK" << std::endl;
}
