// A multi thread, multi stream version of the previous tests
// This will ensure we can control the CUDA streaming of the 
// PyTorch execution
// Memory allocation investigation can be a secondary target.


#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include <sys/prctl.h>
#include "../testBase.h"
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#include <nvToolsExt.h>
#endif

#include "PhysicsTools/PyTorch/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using std::cout;
using std::endl;
using std::exception;

constexpr bool doValidation = true;

// Automatic translation of alpaka platform to torch constants. Embryon of PhysicsTools/PyTorch/interface/config.h
// We rely on HeterogeneousCore/AlpakaInterface/interface/config.h for filtering the defines and assume one and only
// one macro is defined among:
// ALPAKA_ACC_GPU_CUDA_ENABLED
// ALPAKA_ACC_GPU_HIP_ENABLED
// ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
// ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
// The threading model of the CPU versions will have to be investigated.

class NVTXScopedRange {
public:
  NVTXScopedRange(const char * msg) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    id_ = nvtxRangeStartA(msg);
#endif
  }
  
  void end() {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    if (active_) {
      active_ = false;
      nvtxRangeEnd(id_);
    }
#endif
  }
  
  ~NVTXScopedRange() { end(); }
private:
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  nvtxRangeId_t id_;
  bool active_ = true;
#endif
};

struct SimpleDnnSum: torch::nn::Module {
  torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    return A + B;
  }
};

class testTorchFromBufferModelEval : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTorchFromBufferModelEval);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTorchFromBufferModelEval);

using HostBuffer = alpaka::BufCpu<uint8_t, alpaka_common::Dim1D, alpaka_common::Idx>;

/*
 * Demonstration of interoperability between CUDA and Torch C++ API using 
 * pinned memory.
 *
 * Using the ENABLE_ERROR variable a change in the result (CUDA) can be
 * introduced through its respective Torch tensor. This will also affect
 * the copied data from GPU to CPU, resulting in an error during assert
 * checks at the end
 */
// from https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor

//__global__ void vector_add_kernel(int* a, int* b, int* c, int N) {
//  // Calculate global thread ID
//  int t_id = (blockDim.x * blockIdx.x) + threadIdx.x;
//
//  // Check boundry
//  if (t_id < N) {
//    c[t_id] = a[t_id] + b[t_id];
//  }
//}
//
//void vector_add(int* a, int* b, int* c, int N, int cuda_grid_size, int cuda_block_size) {
//  vector_add_kernel<<<cuda_grid_size, cuda_block_size>>>(a, b, c, N);
//  cudaGetLastError();
//}

// bool ENABLE_ERROR = true;
// We take the model as non consr as the forward function is a non const one.
void testTorchFromBufferModelEvalSinglePass(SimpleDnnSum& model, const HostBuffer & a_cpu,  const HostBuffer & b_cpu, HostBuffer & c_cpu, 
        const int N, const size_t bytes, const size_t thread, const size_t iteration, ALPAKA_ACCELERATOR_NAMESPACE::Queue & queue) {
  // Declare GPU memory pointers
  //int *a_gpu, *b_gpu, *c_gpu;

  NVTXScopedRange allocRange("GPU memory allocation");
  // Allocate memory on the device
  cout << "T" << thread << " I" << iteration << " Allocating memory for vectors on GPU" << endl;
  auto a_gpu = alpaka::allocAsyncBuf<uint8_t, uint32_t>(queue, alpaka_common::Vec1D{bytes});
  auto b_gpu = alpaka::allocAsyncBuf<uint8_t, uint32_t>(queue, alpaka_common::Vec1D{bytes});
  auto c_gpu = alpaka::allocAsyncBuf<uint8_t, uint32_t>(queue, alpaka_common::Vec1D{bytes});
  allocRange.end();
  
  
  NVTXScopedRange memcpyRange("Memcpy host to dev");
  // Copy data from the host to the device (CPU -> GPU)
  cout << "T" << thread << " I" << iteration << " Transfering vectors from CPU to GPU" << endl;
  alpaka::memcpy(queue, a_gpu, a_cpu);
  alpaka::memcpy(queue, b_gpu, b_cpu);
  memcpyRange.end();
  
  // Specify threads per CUDA block (CTA), her 2^10 = 1024 threads
  //int NUM_THREADS = 1 << 10;

  // CTAs per grid
  //int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Call CUDA kernel
  //cout << "Running CUDA kernels" << endl;
  //vector_add(a_gpu, b_gpu, c_gpu, N, NUM_BLOCKS, NUM_THREADS);

  NVTXScopedRange inferenceRange((std::string("Inference thread ") + std::to_string(thread)).c_str());
  try {
    // Convert pinned memory on GPU to Torch tensor on GPU
    auto options = torch::TensorOptions().dtype(torch::kInt)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      .device(torch_common::kDeviceType, alpaka::getDev(queue).getNativeHandle())
#endif
      .pinned_memory(true);
    cout << "T" << thread << " I" << iteration << " Converting vectors and result to Torch tensors on GPU" << endl;
    torch::Tensor a_gpu_tensor = torch::from_blob(a_gpu.data(), {N}, options);
    torch::Tensor b_gpu_tensor = torch::from_blob(b_gpu.data(), {N}, options);

    cout << "T" << thread << " I" << iteration << " Running torch inference" << endl;
    std::vector<torch::jit::IValue> inputs{a_gpu_tensor, b_gpu_tensor};
    torch::from_blob(c_gpu.data(), {N}, options) = model.forward(a_gpu_tensor, b_gpu_tensor);

    //CPPUNIT_ASSERT(c_gpu_tensor.equal(output));
  } catch (exception& e) {
    cout << e.what() << endl;
    CPPUNIT_ASSERT(false);
  }
  inferenceRange.end();

  NVTXScopedRange memcpyBackRange("Memcpy dev to host");
  // Copy memory from device and also synchronize (implicitly)
  cout << "T" << thread << " I" << iteration << " Synchronizing CPU and GPU. Copying result from GPU to CPU" << endl;
  alpaka::memcpy(queue, c_cpu, c_gpu);
  memcpyBackRange.end();
  
  if constexpr (doValidation) {
    NVTXScopedRange validationRange("Validation");
    // Verify the result on the CPU
    alpaka::wait(queue);
    cout << "T" << thread << " I" << iteration << " Verifying result on CPU" << endl;
    for (int i = 0; i < N; ++i) {
      CPPUNIT_ASSERT_MESSAGE("ERROR: Mismatch in verification", c_cpu.data()[i] == a_cpu.data()[i] + b_cpu.data()[i]);
    }
    validationRange.end();
  }
}

void testTorchFromBufferModelEval::test() {
  if (prctl(PR_SET_NAME, "test::Main", 0, 0, 0))
    printf ("Warning: Could not set thread name: %s\n", strerror(errno));

  cout << "ALPAKA Platform info:" << endl;
  int idx = 0;
  try {
    for(;;) {
      alpaka::Platform<alpaka::DevCpu> platformHost;
      alpaka::DevCpu host = alpaka::getDevByIdx(platformHost, idx);
      cout << "Host[" << idx++ << "]:   " << alpaka::getName(host) << endl;
    }
  } catch (...) {}
  ALPAKA_ACCELERATOR_NAMESPACE::Platform platform;
  auto alpakaDevices = alpaka::getDevs(platform);
  idx=0;
  for (const auto& d: alpakaDevices) {
      cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << endl;
  }
  const auto & alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto & alpakaDevice = alpakaDevices[0];
  
  cout << "Will create torch device with type=" << torch_common::kDeviceType <<
          " and native handle=" << alpakaDevice.getNativeHandle() << endl;
  torch::Device torchDevice(torch_common::kDeviceType, alpakaDevice.getNativeHandle());
  SimpleDnnSum model;
  
  cout << "Setting the torch thread numbers to 1" << endl
        << "Before:" << endl
        << at::get_parallel_info();
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  cout << "After:" << endl
       << at::get_parallel_info();
  cout << "Loading model..." << endl;
  
  // We need to set the device index to 0 (or a valid value) as leaving it to default (-1) leads to a 
  // bug when setting the cuda stream (-1 is used as an array index without resolving back to 
  // real index (probably).

  // Setup array, here 2^16 = 65536 items
  const int N = 1 << 20;
  size_t bytes = N * sizeof(int);

  // Allocate pinned memory for the pointers
  // The memory will be accessible from both CPU and GPU
  // without the requirements to copy data from one device
  // to the other
  cout << "Allocating memory for vectors on CPU" << endl;
  auto a_cpu = alpaka::allocMappedBuf<ALPAKA_ACCELERATOR_NAMESPACE::Platform, uint8_t, uint32_t>(alpakaHost,platform,alpaka_common::Vec1D{bytes});
  auto b_cpu = alpaka::allocMappedBuf<ALPAKA_ACCELERATOR_NAMESPACE::Platform, uint8_t, uint32_t>(alpakaHost,platform,alpaka_common::Vec1D{bytes});

  // Init vectors
  cout << "Populating vectors with random integers" << endl;
  for (int i = 0; i < N; ++i) {
    a_cpu[i] = rand() % 100;
    b_cpu[i] = rand() % 100;
  }
  
  size_t threadCount = 10;
  std::vector<std::thread> threads;
  for (size_t t=0; t<threadCount; ++t) {
    threads.emplace_back([&, t]{
      char threadName[15];
      snprintf(threadName, 15, "test::%ld", t);
      if (prctl(PR_SET_NAME, threadName, 0, 0, 0))
        printf ("Warning: Could not set thread name: %s\n", strerror(errno));
      cout << "Thread " << t << ": allocating CUDA stream and result buffer" << endl;
      ALPAKA_ACCELERATOR_NAMESPACE::Queue queue{alpakaDevice};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      c10::cuda::CUDAStream torchStream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), torchDevice.index());
      c10::cuda::setCurrentCUDAStream(torchStream);
#endif

      auto c_cpu  = alpaka::allocMappedBuf<ALPAKA_ACCELERATOR_NAMESPACE::Platform, uint8_t, uint32_t>(alpakaHost,platform,alpaka_common::Vec1D{bytes});
      // Get a pyTorch style cuda stream, device is captured from above.
      for (size_t i=0; i<10; ++i)
        testTorchFromBufferModelEvalSinglePass(model, a_cpu, b_cpu, c_cpu, N, bytes, t, i, queue);
      alpaka::wait(queue);
      cout << "Thread " << t << " Test loop complete." << endl;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
#endif
      cout << "Thread " << t << " Stream reset." << endl;
    });
  }
  for (auto &t: threads) t.join();
  cout << "Threads done." << endl;
  // Fixme: free mempory in case of exceptions...
}
