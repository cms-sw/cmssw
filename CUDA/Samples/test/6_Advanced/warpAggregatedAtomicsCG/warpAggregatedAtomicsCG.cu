/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define NUM_ELEMS 10000000
#define NUM_THREADS_PER_BLOCK 512

// warp-aggregated atomic increment
__device__ int atomicAggInc(int *counter)
{
    cg::coalesced_group active = cg::coalesced_threads();

    int mask = active.ballot(1);
    // select the leader
    int leader = __ffs(mask) - 1;

    // leader does the update
    int res = 0;
    if (active.thread_rank() == leader)
    {
        res = atomicAdd(counter, __popc(mask));
    }

    // broadcast result
    res = active.shfl(res, leader);

    // each thread computes its own value
    return res + __popc(mask & ((1 << active.thread_rank()) - 1));
}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = id; i < n; i += gridDim.x*blockDim.x)
    {
        if(src[i] > 0)
            dst[atomicAggInc(nres)] = src[i];
    }
}


int main(int argc, char **argv)
{
    int *data_to_filter, *filtered_data, nres=0;
    int *d_data_to_filter, *d_filtered_data, *d_nres;

    data_to_filter = (int *) malloc(sizeof(int)*NUM_ELEMS);

    // Generate input data.
    for (int i=0; i < NUM_ELEMS; i++)
    {
        data_to_filter[i] = rand() % 20;
    }

    findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaMalloc(&d_data_to_filter, sizeof(int)*NUM_ELEMS));
    checkCudaErrors(cudaMalloc(&d_filtered_data, sizeof(int)*NUM_ELEMS));
    checkCudaErrors(cudaMalloc(&d_nres, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_data_to_filter, data_to_filter, sizeof(int)*NUM_ELEMS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_nres, 0, sizeof(int)));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS/NUM_THREADS_PER_BLOCK) + 1, 1, 1);

    filter_arr<<<dimGrid, dimBlock>>>(d_filtered_data, d_nres, d_data_to_filter, NUM_ELEMS);

    checkCudaErrors(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));

    filtered_data = (int *) malloc(sizeof(int)*nres);

    checkCudaErrors(cudaMemcpy(filtered_data, d_filtered_data, sizeof(int)*nres, cudaMemcpyDeviceToHost));

    int *host_filtered_data = (int *) malloc(sizeof(int)*NUM_ELEMS);

    // Generate host output with host filtering code.
    int host_flt_count=0;
    for (int i=0; i < NUM_ELEMS; i++)
    {
        if (data_to_filter[i] > 0)
        {
            host_filtered_data[host_flt_count++] = data_to_filter[i];
        }
    }

    printf("\nWarp Aggregated Atomics %s \n",  host_flt_count == nres ? "PASSED" : "FAILED");

    checkCudaErrors(cudaFree(d_data_to_filter));
    checkCudaErrors(cudaFree(d_filtered_data));
    checkCudaErrors(cudaFree(d_nres));
    free(data_to_filter); 
    free(filtered_data);
    free(host_filtered_data);
}

