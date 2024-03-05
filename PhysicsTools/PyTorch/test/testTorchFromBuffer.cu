#include <cuda_runtime.h>
#include <torch/torch.h>
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>

using std::cout;
using std::endl;
using std::exception;

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

__global__ void vector_add_kernel(int* a, int* b, int* c, int N)
{
    // Calculate global thread ID
    int t_id = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Check boundry
    if (t_id < N)
    {
        c[t_id] = a[t_id] + b[t_id];
    }
}

void vector_add(int* a, int* b, int* c, int N, int cuda_grid_size, int cuda_block_size)
{
    vector_add_kernel << <cuda_grid_size, cuda_block_size >> > (a, b, c, N);
    cudaGetLastError();
}


bool ENABLE_ERROR = false;

int main(int argc, const char* argv[])
{
    // Setup array, here 2^16 = 65536 items
    const int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    // Declare pinned memory pointers
    int* a_cpu, * b_cpu, * c_cpu;

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
    for (int i = 0; i < N; ++i)
    {
        a_cpu[i] = rand() % 100;
        b_cpu[i] = rand() % 100;
    }

    // Declare GPU memory pointers
    int* a_gpu, * b_gpu, * c_gpu;

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
    int NUM_THREADS = 1 << 10;

    // CTAs per grid
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Call CUDA kernel
    cout << "Running CUDA kernels" << endl;
    vector_add(a_gpu, b_gpu, c_gpu, N, NUM_BLOCKS, NUM_THREADS);

    try
    {
        // Convert pinned memory on GPU to Torch tensor on GPU
        auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0).pinned_memory(true);
        cout << "Converting vectors and result to Torch tensors on GPU" << endl;
        torch::Tensor a_gpu_tensor = torch::from_blob(a_gpu, { N }, options);
        torch::Tensor b_gpu_tensor = torch::from_blob(b_gpu, { N }, options);
        torch::Tensor c_gpu_tensor = torch::from_blob(c_gpu, { N }, options);

        cout << "Verifying result using Torch tensors" << endl;
        if (ENABLE_ERROR)
        {
            /*
            TEST
            Change the value of the result should result in two things:
             - the GPU memory will be modified
             - the CPU test later on (after the GPU memory is copied to the CPU side) should fail
            */
            cout << "ERROR GENERATION ENABLED! Application will crash during verification of results" << endl;
            cout << "Changing result first element from " << c_gpu_tensor[0];
            c_gpu_tensor[0] = 99999999;
            cout << " to " << c_gpu_tensor[0] << endl;
        }
        else
        {
            assert(c_gpu_tensor.equal(a_gpu_tensor.add(b_gpu_tensor)) == true);
        }
    }
    catch (exception& e)
    {
        cout << e.what() << endl;

        cudaFreeHost(a_cpu);
        cudaFreeHost(b_cpu);
        cudaFreeHost(c_cpu);

        cudaFree(a_gpu);
        cudaFree(b_gpu);
        cudaFree(c_gpu);

        return 1;
    }

    // Copy memory to device and also synchronize (implicitly)
    cout << "Synchronizing CPU and GPU. Copying result from GPU to CPU" << endl;
    cudaMemcpy(c_cpu, c_gpu, bytes, cudaMemcpyDeviceToHost);

    // Verify the result on the CPU
    cout << "Verifying result on CPU" << endl;
    for (int i = 0; i < N; ++i)
    {
        assert(c_cpu[i] == a_cpu[i] + b_cpu[i]);
    }

    cudaFreeHost(a_cpu);
    cudaFreeHost(b_cpu);
    cudaFreeHost(c_cpu);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return 0;
}
