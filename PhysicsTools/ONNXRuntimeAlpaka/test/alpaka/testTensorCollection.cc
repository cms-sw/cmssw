#include <cstdint>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <Eigen/Core>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/PackKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest {

  using cms::Ort::alpakatools::Layout;
  using cms::Ort::alpakatools::TensorCollection;
  using cms::Ort::alpakatools::TensorSlice;

  using Matrix23 = Eigen::Matrix<float, 2, 3>;
  GENERATE_SOA_LAYOUT(TestLayout,
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, v),
                      SOA_EIGEN_COLUMN(Matrix23, m1),
                      SOA_EIGEN_COLUMN(Matrix23, m2),
                      SOA_COLUMN(double, d),
                      SOA_COLUMN(uint8_t, u),
                      SOA_SCALAR(float, s))
  using TestSoA = TestLayout<>;
  using TestDeviceCollection = PortableCollection<Device, TestSoA>;
  using TestHostCollection = PortableHostCollection<TestSoA>;

  // value of element (r, c) of the matrix k of row i
  constexpr float value(int i, int k, int r, int c) { return i * 1000 + k * 100 + r * 10 + c; }

  class TestTensorCollection : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestTensorCollection);
    CPPUNIT_TEST(testSlice);
    CPPUNIT_TEST(testShapes);
    CPPUNIT_TEST(testFeatureMajor);
    CPPUNIT_TEST(testDuplicateName);
    CPPUNIT_TEST(testChangeOrder);
    CPPUNIT_TEST(testMatch);
    CPPUNIT_TEST(testPack);
    CPPUNIT_TEST_SUITE_END();

  public:
    void setUp() override {
      auto devices = alpaka::getDevs(Platform{});
      CPPUNIT_ASSERT(not devices.empty());
      queue_.emplace(devices[0]);
    }
    void tearDown() override { queue_.reset(); }

    void testSlice();
    void testShapes();
    void testFeatureMajor();
    void testDuplicateName();
    void testChangeOrder();
    void testMatch();
    void testPack();

  private:
    std::optional<Queue> queue_;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTensorCollection);

  void TestTensorCollection::testSlice() {
    auto check = [](TensorSlice slice, uint32_t total, uint32_t offset, uint32_t size) {
      auto bounds = slice.resolve(total);
      CPPUNIT_ASSERT_EQUAL(offset, bounds.offset);
      CPPUNIT_ASSERT_EQUAL(size, bounds.size);
    };
    check(TensorSlice{}, 35, 0, 35);
    check(TensorSlice{}, 0, 0, 0);
    check(TensorSlice{0, 32}, 35, 0, 32);
    check(TensorSlice{1, 32}, 35, 32, 3);
    check(TensorSlice{0, 64}, 35, 0, 35);
    check(TensorSlice{1, 32}, 64, 32, 32);
    check(TensorSlice{3, 32}, 0, 0, 0);
    CPPUNIT_ASSERT_THROW(TensorSlice(2, 32).resolve(64), cms::Exception);
    CPPUNIT_ASSERT_THROW(TensorSlice(0, 0).resolve(35), cms::Exception);
  }

  void TestTensorCollection::testShapes() {
    using V = std::vector<int64_t>;
    TestDeviceCollection collection(*queue_, 35);
    auto records = collection.view().records();
    // columns are padded to 128 bytes: 64 floats for 35 elements
    TensorCollection<Queue> tensors;
    tensors.add<TestSoA>("xyz", records.x(), records.y(), records.z());
    tensors.add<TestSoA>("x", records.x());
    tensors.add<TestSoA>("v", records.v());
    tensors.add<TestSoA>("m", records.m1(), records.m2());
    tensors.add<TestSoA>("s", records.s());
    tensors.add<TestSoA>("y", TensorSlice{1, 32}, records.y());

    CPPUNIT_ASSERT(tensors[0].sizes() == V({35, 3}));
    CPPUNIT_ASSERT(tensors[0].strides() == V({1, 64}));
    CPPUNIT_ASSERT(not tensors[0].is_contiguous());
    CPPUNIT_ASSERT(tensors[1].sizes() == V({35, 1}));
    CPPUNIT_ASSERT(tensors[1].is_contiguous());
    CPPUNIT_ASSERT(tensors[2].sizes() == V({35, 3}));
    CPPUNIT_ASSERT(tensors[2].strides() == V({1, 64}));
    CPPUNIT_ASSERT(tensors[3].sizes() == V({35, 2, 2, 3}));
    CPPUNIT_ASSERT(tensors[3].strides() == V({1, 64 * 6, 64, 64 * 2}));
    CPPUNIT_ASSERT(tensors[4].sizes() == V({1}));
    CPPUNIT_ASSERT(tensors[4].is_scalar());
    CPPUNIT_ASSERT(tensors[4].is_contiguous());
    CPPUNIT_ASSERT(tensors[5].sizes() == V({3, 1}));
    CPPUNIT_ASSERT(tensors[5].data() == static_cast<void*>(collection.view().y().data() + 32));
    CPPUNIT_ASSERT(not tensors[5].full_range());

    // tensors of different SoAs, with different sizes
    TestDeviceCollection other(*queue_, 7);
    tensors.add<TestSoA>("other", other.view().records().x(), other.view().records().y());
    CPPUNIT_ASSERT(tensors[6].sizes() == V({7, 2}));
    CPPUNIT_ASSERT(tensors[6].strides() == V({1, 32}));
  }

  void TestTensorCollection::testFeatureMajor() {
    using V = std::vector<int64_t>;
    TestDeviceCollection collection(*queue_, 35);
    auto records = collection.view().records();
    TensorCollection<Queue> tensors;
    tensors.add<TestSoA>("xyz", records.x(), records.y(), records.z());
    tensors.add<TestSoA>("d", records.d());
    tensors.add<TestSoA>("u", records.u());
    tensors.add<TestSoA>("m", records.m1(), records.m2());
    tensors.add<TestSoA>("s", records.s());
    tensors.add<TestSoA>("slice", TensorSlice{0, 32}, records.x());
    for (auto name : {"xyz", "d", "u", "m"})
      tensors.set_layout(name, Layout::FeatureMajor);
    // the padded size depends on the size of the elements
    CPPUNIT_ASSERT(tensors[0].shape() == V({3, 64}));
    CPPUNIT_ASSERT(tensors[0].is_contiguous());
    CPPUNIT_ASSERT(tensors[1].shape() == V({1, 48}));
    CPPUNIT_ASSERT(tensors[2].shape() == V({1, 128}));
    CPPUNIT_ASSERT(tensors[3].shape() == V({12, 64}));
    // not supported for scalars and mini-batches
    CPPUNIT_ASSERT_THROW(tensors.set_layout("s", Layout::FeatureMajor), cms::Exception);
    CPPUNIT_ASSERT_THROW(tensors.set_layout("slice", Layout::FeatureMajor), cms::Exception);
    CPPUNIT_ASSERT_THROW(tensors.set_layout("missing", Layout::FeatureMajor), cms::Exception);
  }

  void TestTensorCollection::testDuplicateName() {
    TestDeviceCollection collection(*queue_, 35);
    auto records = collection.view().records();
    TensorCollection<Queue> tensors;
    // the same name twice
    tensors.add<TestSoA>("x", records.x());
    CPPUNIT_ASSERT_THROW(tensors.add<TestSoA>("x", records.y()), cms::Exception);
  }

  void TestTensorCollection::testChangeOrder() {
    TestDeviceCollection collection(*queue_, 35);
    auto records = collection.view().records();
    TensorCollection<Queue> tensors;
    tensors.add<TestSoA>("a", records.x());
    tensors.add<TestSoA>("b", records.y());
    tensors.change_order({"b", "a"});
    CPPUNIT_ASSERT_EQUAL(std::string("b"), tensors.name(0));
    CPPUNIT_ASSERT(tensors[0].data() == static_cast<void*>(collection.view().y().data()));
    CPPUNIT_ASSERT_THROW(tensors.change_order({"a", "a"}), cms::Exception);
    CPPUNIT_ASSERT_THROW(tensors.change_order({"a", "c"}), cms::Exception);
    CPPUNIT_ASSERT_THROW(tensors.change_order({"a"}), cms::Exception);
    CPPUNIT_ASSERT_EQUAL(std::string("b"), tensors.name(0));
  }

  void TestTensorCollection::testMatch() {
    using I = std::vector<size_t>;
    TestDeviceCollection collection(*queue_, 35);
    auto records = collection.view().records();
    TensorCollection<Queue> tensors;
    tensors.add<TestSoA>("mask", records.u());
    tensors.add<TestSoA>("particles", records.x(), records.y(), records.z());
    // by name, independently of the order
    CPPUNIT_ASSERT(tensors.match({"particles", "mask"}, "input", "test.onnx") == I({1, 0}));
    CPPUNIT_ASSERT(tensors.match({"mask", "particles"}, "input", "test.onnx") == I({0, 1}));
    // by position, if no name matches
    CPPUNIT_ASSERT(tensors.match({"input_0", "input_1"}, "input", "test.onnx") == I({0, 1}));
    // a partial match is an error
    CPPUNIT_ASSERT_THROW(tensors.match({"particles", "masks"}, "input", "test.onnx"), cms::Exception);
    // the number of tensors must match the model
    CPPUNIT_ASSERT_THROW(tensors.match({"particles"}, "input", "test.onnx"), cms::Exception);
    CPPUNIT_ASSERT_THROW(tensors.match({"particles", "mask", "extra"}, "input", "test.onnx"), cms::Exception);
  }

  void TestTensorCollection::testPack() {
    constexpr int n = 35;
    Queue& queue = *queue_;
    TestHostCollection host(cms::alpakatools::host(), n);
    for (int i = 0; i < n; ++i) {
      host.view()[i].x() = value(i, 0, 0, 0);
      host.view()[i].y() = value(i, 1, 0, 0);
      host.view()[i].z() = value(i, 2, 0, 0);
      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
          host.view()[i].m1()(r, c) = value(i, 0, r, c);
          host.view()[i].m2()(r, c) = value(i, 1, r, c);
        }
    }
    TestDeviceCollection device(queue, n);
    alpaka::memcpy(queue, device.buffer(), host.buffer());
    auto records = device.view().records();

    // pack the second mini-batch of 16 matrices into a [batch, 2, 2, 3] row-major buffer
    TensorCollection<Queue> tensors;
    tensors.add<TestSoA>("m", TensorSlice{1, 16}, records.m1(), records.m2());
    tensors.add<TestSoA>("xyz", TensorSlice{2, 16}, records.x(), records.y(), records.z());
    auto const& m = tensors[0];
    auto const& xyz = tensors[1];
    CPPUNIT_ASSERT_EQUAL(16, m.batch_size());
    CPPUNIT_ASSERT_EQUAL(3, xyz.batch_size());

    auto buffer = cms::alpakatools::make_device_buffer<std::byte[]>(queue, m.shape_bytes());
    ort::detail::pack(queue, m.pack_descriptor(), static_cast<std::byte const*>(m.data()), buffer.data());
    auto buffer_xyz = cms::alpakatools::make_device_buffer<std::byte[]>(queue, xyz.shape_bytes());
    ort::detail::pack(queue, xyz.pack_descriptor(), static_cast<std::byte const*>(xyz.data()), buffer_xyz.data());
    auto packed = cms::alpakatools::make_host_buffer<float[]>(queue, m.shape_bytes() / sizeof(float));
    auto packed_xyz = cms::alpakatools::make_host_buffer<float[]>(queue, xyz.shape_bytes() / sizeof(float));
    alpaka::memcpy(queue,
                   packed,
                   cms::alpakatools::make_device_view(
                       queue, reinterpret_cast<float*>(buffer.data()), m.shape_bytes() / sizeof(float)));
    alpaka::memcpy(queue,
                   packed_xyz,
                   cms::alpakatools::make_device_view(
                       queue, reinterpret_cast<float*>(buffer_xyz.data()), xyz.shape_bytes() / sizeof(float)));
    alpaka::wait(queue);
    for (int b = 0; b < 16; ++b)
      for (int k = 0; k < 2; ++k)
        for (int r = 0; r < 2; ++r)
          for (int c = 0; c < 3; ++c)
            CPPUNIT_ASSERT_EQUAL(value(16 + b, k, r, c), packed[((b * 2 + k) * 2 + r) * 3 + c]);
    for (int b = 0; b < 3; ++b)
      for (int k = 0; k < 3; ++k)
        CPPUNIT_ASSERT_EQUAL(value(32 + b, k, 0, 0), packed_xyz[b * 3 + k]);

    // zero the same mini-batch, then unpack the buffer into it again
    ort::detail::zero(queue, m.pack_descriptor(), static_cast<std::byte*>(m.data()));
    alpaka::memcpy(queue, host.buffer(), device.buffer());
    alpaka::wait(queue);
    for (int i = 0; i < n; ++i)
      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
          bool zeroed = (i >= 16 and i < 32);
          CPPUNIT_ASSERT_EQUAL(zeroed ? 0.f : value(i, 0, r, c), host.view()[i].m1()(r, c));
          CPPUNIT_ASSERT_EQUAL(zeroed ? 0.f : value(i, 1, r, c), host.view()[i].m2()(r, c));
        }
    ort::detail::unpack(queue, m.pack_descriptor(), buffer.data(), static_cast<std::byte*>(m.data()));
    alpaka::memcpy(queue, host.buffer(), device.buffer());
    alpaka::wait(queue);
    for (int i = 0; i < n; ++i)
      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
          CPPUNIT_ASSERT_EQUAL(value(i, 0, r, c), host.view()[i].m1()(r, c));
          CPPUNIT_ASSERT_EQUAL(value(i, 1, r, c), host.view()[i].m2()(r, c));
        }
    // the columns that do not belong to the tensor are untouched
    for (int i = 0; i < n; ++i)
      CPPUNIT_ASSERT_EQUAL(value(i, 0, 0, 0), host.view()[i].x());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest
