#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include "testBase.h"

using std::cout;
using std::endl;
using std::exception;

class testTorchFromBufferModelEval : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testTorchFromBufferModelEval);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTorchFromBufferModelEval);

std::string testTorchFromBufferModelEval::pyScript() const { return "create_dnn_largeinput.py"; }
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

__global__ void vector_add_kernel(int* a, int* b, int* c, int N) {
  // Calculate global thread ID
  int t_id = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Check boundry
  if (t_id < N) {
    c[t_id] = a[t_id] + b[t_id];
  }
}

void vector_add(int* a, int* b, int* c, int N, int cuda_grid_size, int cuda_block_size) {
  vector_add_kernel<<<cuda_grid_size, cuda_block_size>>>(a, b, c, N);
  cudaGetLastError();
}

// bool ENABLE_ERROR = true;

//int oldMain(int argc, const char* argv[])
void testTorchFromBufferModelEval::test() {
  // Setup array, here 2^16 = 65536 items
  const int N = 1 << 16;
  size_t bytes = N * sizeof(int);

  // Declare pinned memory pointers
  int *a_cpu, *b_cpu, *c_cpu;

  // Allocate pinned memory for the pointers
  // The memory will be accessible from both CPU and GPU
  // without the requirements to copy data from one device
  // to the other
  cout << "Allocating memory for vectors on CPU" << endl;
  cudaMallocHost(&a_cpu, bytes);
  cudaMallocHost(&b_cpu, bytes);
  cudaMallocHost(&c_cpu, bytes);

  // Init vectors
  cout << "Populating vectors with random integers" << endl;
  for (int i = 0; i < N; ++i) {
    a_cpu[i] = rand() % 100;
    b_cpu[i] = rand() % 100;
  }

  // Declare GPU memory pointers
  int *a_gpu, *b_gpu, *c_gpu;

  // Allocate memory on the device
  cout << "Allocating memory for vectors on GPU" << endl;
  cudaMalloc(&a_gpu, bytes);
  cudaMalloc(&b_gpu, bytes);
  cudaMalloc(&c_gpu, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cout << "Transfering vectors from CPU to GPU" << endl;
  cudaMemcpy(a_gpu, a_cpu, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b_cpu, bytes, cudaMemcpyHostToDevice);

  // Specify threads per CUDA block (CTA), her 2^10 = 1024 threads
  //int NUM_THREADS = 1 << 10;

  // CTAs per grid
  //int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Call CUDA kernel
  //cout << "Running CUDA kernels" << endl;
  //vector_add(a_gpu, b_gpu, c_gpu, N, NUM_BLOCKS, NUM_THREADS);

  // Load the TorchScript model
  std::string model_path = dataPath_ + "/simple_dnn_largeinput.pt";

  torch::jit::script::Module model;
  torch::Device device(torch::kCUDA);
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(model_path);
    model.to(device);

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }

  try {
    // Convert pinned memory on GPU to Torch tensor on GPU
    auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0).pinned_memory(true);
    cout << "Converting vectors and result to Torch tensors on GPU" << endl;
    torch::Tensor a_gpu_tensor = torch::from_blob(a_gpu, {N}, options);
    torch::Tensor b_gpu_tensor = torch::from_blob(b_gpu, {N}, options);

    cout << "Verifying result using Torch tensors" << endl;
    std::vector<torch::jit::IValue> inputs{a_gpu_tensor, b_gpu_tensor};
    // Not fully understood but std::move() is needed
    // https://stackoverflow.com/questions/71790378/assign-memory-blob-to-py-torch-output-tensor-c-api 
    torch::from_blob(c_gpu, {N}, options) = model.forward(inputs).toTensor();

    //CPPUNIT_ASSERT(c_gpu_tensor.equal(output));
  } catch (exception& e) {
    cout << e.what() << endl;

    cudaFreeHost(a_cpu);
    cudaFreeHost(b_cpu);
    cudaFreeHost(c_cpu);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    CPPUNIT_ASSERT(false);
  }

  // Copy memory to device and also synchronize (implicitly)
  cout << "Synchronizing CPU and GPU. Copying result from GPU to CPU" << endl;
  cudaMemcpy(c_cpu, c_gpu, bytes, cudaMemcpyDeviceToHost);

  // Verify the result on the CPU
  cout << "Verifying result on CPU" << endl;
  for (int i = 0; i < N; ++i) {
    CPPUNIT_ASSERT(c_cpu[i] == a_cpu[i] + b_cpu[i]);
  }

  cudaFreeHost(a_cpu);
  cudaFreeHost(b_cpu);
  cudaFreeHost(c_cpu);

  cudaFree(a_gpu);
  cudaFree(b_gpu);
  cudaFree(c_gpu);
}
