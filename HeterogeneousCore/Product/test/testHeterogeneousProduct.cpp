#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

class testHeterogeneousProduct: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testHeterogeneousProduct);
  CPPUNIT_TEST(testDefault);
  CPPUNIT_TEST(testCPU);
  CPPUNIT_TEST(testGPUMock);
  CPPUNIT_TEST(testGPUCuda);
  CPPUNIT_TEST(testGPUAll);
  CPPUNIT_TEST(testMoveGPUMock);
  CPPUNIT_TEST(testMoveGPUCuda);
  CPPUNIT_TEST(testProduct);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void testDefault();
  void testCPU();
  void testGPUMock();
  void testGPUCuda();
  void testGPUAll();
  void testMoveGPUMock();
  void testMoveGPUCuda();
  void testProduct();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testHeterogeneousProduct);

void testHeterogeneousProduct::testDefault() {
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUMockProduct<int>
                           > prod;

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT_THROW(prod.getProduct<HeterogeneousDevice::kCPU>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
}

void testHeterogeneousProduct::testCPU() {
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUMockProduct<int>
                           > prod{heterogeneous::cpuProduct(5)};

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT_THROW(prod.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
}

void testHeterogeneousProduct::testGPUMock() {
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUMockProduct<int>
                           > prod{heterogeneous::gpuMockProduct(5),
                                  [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUMock>() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
}

void testHeterogeneousProduct::testGPUCuda() {
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUCudaProduct<int>
                           > prod{heterogeneous::gpuCudaProduct(5),
                                  [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == true);

  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);
}

void testHeterogeneousProduct::testGPUAll() {
  // Data initially on CPU
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUMockProduct<int>,
                           heterogeneous::GPUCudaProduct<int>
                           > prod1{heterogeneous::cpuProduct(5)};

  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod1.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT_THROW(prod1.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod1.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // Data initially on GPUMock
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUMockProduct<int>,
                           heterogeneous::GPUCudaProduct<int>
                           > prod2{heterogeneous::gpuMockProduct(5),
                                   [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // Automatic transfer
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // Data initially on GPUCuda
  HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                           heterogeneous::GPUMockProduct<int>,
                           heterogeneous::GPUCudaProduct<int>
                           > prod3{heterogeneous::gpuCudaProduct(5),
                                   [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUCuda) == true);

  CPPUNIT_ASSERT_THROW(prod3.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod3.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(prod3.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT_THROW(prod3.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod3.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);
}


void testHeterogeneousProduct::testMoveGPUMock() {
  // Data initially on GPUMock
  using Type = HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                                        heterogeneous::GPUMockProduct<int>,
                                        heterogeneous::GPUCudaProduct<int>
                                        >;
  Type prod1{heterogeneous::gpuMockProduct(5),
             [](const int& src, int& dst) { dst = src; }};
  Type prod2;

  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod1.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod1.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // move
  prod2 = std::move(prod1);

  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // automatic transfer
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);
}

void testHeterogeneousProduct::testMoveGPUCuda() {
  // Data initially on GPUCuda
  using Type = HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                                        heterogeneous::GPUMockProduct<int>,
                                        heterogeneous::GPUCudaProduct<int>
                                        >;
  Type prod1{heterogeneous::gpuCudaProduct(5),
             [](const int& src, int& dst) { dst = src; }};
  Type prod2;

  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT_THROW(prod1.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod1.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // move
  prod2 = std::move(prod1);

  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == true);

  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);

  // automatic transfer
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);
}

void testHeterogeneousProduct::testProduct() {
  using Type1 = HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                                         heterogeneous::GPUMockProduct<int>
                                         >;
  using Type2 = HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                                         heterogeneous::GPUCudaProduct<int>
                                         >;

  Type1 data1{heterogeneous::cpuProduct(5)};
  Type2 data2{heterogeneous::cpuProduct(10)};

  HeterogeneousProduct prod{};
  CPPUNIT_ASSERT(prod.isNull() == true);
  CPPUNIT_ASSERT(prod.isNonnull() == false);
  CPPUNIT_ASSERT_THROW(prod.get<Type1>(), cms::Exception);

  HeterogeneousProduct prod1{std::move(data1)};
  CPPUNIT_ASSERT(prod1.isNull() == false);
  CPPUNIT_ASSERT(prod1.isNonnull() == true);
  CPPUNIT_ASSERT(prod1.get<Type1>().getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT_THROW(prod1.get<Type2>(), cms::Exception);

  HeterogeneousProduct prod2{std::move(data2)};
  CPPUNIT_ASSERT(prod2.isNull() == false);
  CPPUNIT_ASSERT(prod2.isNonnull() == true);
  CPPUNIT_ASSERT_THROW(prod2.get<Type1>(), cms::Exception);
  CPPUNIT_ASSERT(prod2.get<Type2>().getProduct<HeterogeneousDevice::kCPU>() == 10);

  prod1 = std::move(prod2);
  CPPUNIT_ASSERT_THROW(prod1.get<Type1>(), cms::Exception);
  CPPUNIT_ASSERT(prod1.get<Type2>().getProduct<HeterogeneousDevice::kCPU>() == 10);

  prod = std::move(prod1);
  CPPUNIT_ASSERT(prod.isNull() == false);
  CPPUNIT_ASSERT(prod.isNonnull() == true);
  CPPUNIT_ASSERT_THROW(prod.get<Type1>(), cms::Exception);
  CPPUNIT_ASSERT(prod.get<Type2>().getProduct<HeterogeneousDevice::kCPU>() == 10);
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
