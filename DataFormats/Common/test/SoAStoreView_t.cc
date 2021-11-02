/*
 * SoASoteView_t.cc
 * 
 * A test validating and illustrating the use of SoAStores and Views.
 * This code takes the form of pseudo kernels called directly in order to easily port it
 * to accelerators. 
 */

#include <cassert>
#include <memory>

#include "DataFormats/Common/interface/SoAStore.h"
#include "DataFormats/Common/interface/SoAView.h"
#include "Eigen/Geometry"

generate_SoA_store(SoAHostDevice,
                   // predefined static scalars
                   // size_t size;
                   // size_t alignment;

                   // columns: one value per element
                   SoA_column(double, x),
                   SoA_column(double, y),
                   SoA_column(double, z),
                   SoA_eigenColumn(Eigen::Vector3d, a),
                   SoA_eigenColumn(Eigen::Vector3d, b),
                   SoA_eigenColumn(Eigen::Vector3d, r),
                   // scalars: one value for the whole structure
                   SoA_scalar(const char *, description),
                   SoA_scalar(uint32_t, someNumber));

generate_SoA_store(SoADeviceOnly,
                   SoA_column(uint16_t, color),
                   SoA_column(double, value),
                   SoA_column(double *, py),
                   SoA_column(uint32_t, count),
                   SoA_column(uint32_t, anotherCount));

// A 1 to 1 view of the store (except for unsupported types).
generate_SoA_view(SoAFullDeviceView,
                  SoA_view_store_list(SoA_view_store(SoAHostDevice, soaHD), SoA_view_store(SoADeviceOnly, soaDO)),
                  SoA_view_value_list(SoA_view_value(soaHD, x, x),
                                      SoA_view_value(soaHD, y, y),
                                      SoA_view_value(soaHD, z, z),
                                      SoA_view_value(soaDO, color, color),
                                      SoA_view_value(soaDO, value, value),
                                      SoA_view_value(soaDO, py, py),
                                      SoA_view_value(soaDO, count, count),
                                      SoA_view_value(soaDO, anotherCount, anotherCount),
                                      SoA_view_value(soaHD, description, description),
                                      SoA_view_value(soaHD, someNumber, someNumber)));

// Eigen cross product kernel (on store)
struct crossProduct {
  void operator()(SoAHostDevice soa, size_t i) const {
    auto si = soa[i];
    si.r() = si.a().cross(si.b());
  }
};

// Device-only producer kernel
struct producerKernel {
  void operator()(SoAFullDeviceView soa, size_t i) const {
    auto si = soa[i];
    si.color() &= 0x55 << i % (sizeof(si.color()) - sizeof(char));
    si.value() = sqrt(si.x() * si.x() + si.y() * si.y() + si.z() * si.z());
  }
};

// Device-only consumer with result in host-device area
struct consumerKernel {
  void operator()(SoAFullDeviceView soa, size_t i) const {
    auto si = soa[i];
    si.x() = si.color() * si.value();
  }
};

using AlignedBuffer = std::unique_ptr<std::byte, decltype(std::free) *>;

int main(void) {
  // Non-aligned number of elements to check alignment features.
  constexpr unsigned int numElements = 65537;

  // We target a CUDA-like alignment
  const size_t byteAlignment =
      128;  // The default alignment for SoA (nVidia GPI cache line size, reflected in CUDA memory allocations).

  // Allocate buffer and store on host
  size_t hostDeviceSize = SoAHostDevice::computeDataSize(numElements);
  AlignedBuffer h_buf(reinterpret_cast<std::byte *>(aligned_alloc(byteAlignment, hostDeviceSize)), std::free);
  SoAHostDevice h_soahd(h_buf.get(), numElements, byteAlignment);

  // Alocate buffer, stores and views on the device (single, shared buffer).
  size_t deviceOnlySize = SoADeviceOnly::computeDataSize(numElements);
  AlignedBuffer d_buf(reinterpret_cast<std::byte *>(aligned_alloc(byteAlignment, hostDeviceSize + deviceOnlySize)),
                      std::free);
  SoAHostDevice d_soahd(d_buf.get(), numElements, byteAlignment);
  SoADeviceOnly d_soado(d_soahd.soaMetadata().nextByte(), numElements, byteAlignment);
  SoAFullDeviceView d_soa(d_soahd, d_soado);

  // Assert column alignments
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.x()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.y()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.z()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.a()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.b()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(h_soahd.r()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(&h_soahd.description()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(&h_soahd.someNumber()) % h_soahd.soaMetadata().byteAlignment());

  assert(0 == reinterpret_cast<uintptr_t>(d_soahd.x()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soahd.y()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soahd.z()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soahd.a()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soahd.b()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soahd.r()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(&d_soahd.description()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(&d_soahd.someNumber()) % h_soahd.soaMetadata().byteAlignment());

  assert(0 == reinterpret_cast<uintptr_t>(d_soado.color()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soado.value()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soado.py()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soado.count()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soado.anotherCount()) % h_soahd.soaMetadata().byteAlignment());

  // Views should get the same alignment as the stores they refer to
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.x()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.y()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.z()) % h_soahd.soaMetadata().byteAlignment());
  // Limitation of views: we have to get scalar member addresses via metadata.
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.soaMetadata().addressOf_description()) %
                  h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.soaMetadata().addressOf_someNumber()) %
                  h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.color()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.value()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.py()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.count()) % h_soahd.soaMetadata().byteAlignment());
  assert(0 == reinterpret_cast<uintptr_t>(d_soa.anotherCount()) % h_soahd.soaMetadata().byteAlignment());

  // Initialize and fill the host buffer
  std::memset(h_soahd.soaMetadata().data(), 0, hostDeviceSize);
  for (size_t i = 0; i < numElements; ++i) {
    auto si = h_soahd[i];
    si.x() = si.a()(0) = si.b()(2) = 1.0 * i + 1.0;
    si.y() = si.a()(1) = si.b()(1) = 2.0 * i;
    si.z() = si.a()(2) = si.b()(0) = 3.0 * i - 1.0;
  }
  h_soahd.someNumber() = numElements + 2;

  // Push to "device"
  std::memcpy(d_buf.get(), h_buf.get(), hostDeviceSize);

  // Process on "device"
  for (size_t i = 0; i < numElements; i++)
    crossProduct()(d_soahd, i);

  // Initialize the device only part
  std::memset(d_soado.soaMetadata().data(), 0xFF, d_soado.soaMetadata().byteSize());

  // Produce to the device only area
  for (size_t i = 0; i < numElements; i++)
    producerKernel()(d_soa, i);

  // Consume the device only area and generate a result on the host-device area
  for (size_t i = 0; i < numElements; i++)
    consumerKernel()(d_soa, i);

  // Get result back
  std::memcpy(h_buf.get(), d_buf.get(), hostDeviceSize);

  // Wait and validate.
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
  std::cout << "OK" << std::endl;
}
