/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 *
 * This sample is a simple code that illustrates basic usage of
 * cooperative groups within the thread block. The code launches a single
 * thread block, creates a cooperative group of all threads in the block,
 * and a set of tiled partition cooperative groups. For each, it uses a
 * generic reduction function to calculate the sum of all the ranks in
 * that group. In each case the result is printed, together with the
 * expected answer (which is calculated using the analytical formula
 * (n-1)*n)/2, noting that the ranks start at zero).
 *
 */

#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;


/**
 * CUDA device function
 * 
 * calculates the sum of val across the group g. The workspace array, x, 
 * must be large enough to contain g.size() integers.
 */
__device__ int sumReduction(thread_group g, int *x, int val) 
{ 
    // rank of this thread in the group 
    int lane = g.thread_rank(); 

    // for each iteration of this loop, the number of threads active in the
    // reduction, i, is halved, and each active thread (with index [lane])
    // performs a single summation of it's own value with that
    // of a "partner" (with index [lane+i]). 
    for (int i = g.size()/2; i > 0; i /= 2) 
        { 

            // store value for this thread in temporary array
            x[lane] = val;

            // synchronize all threads in group
            g.sync();


            if(lane<i)
                // active threads perform summation of their value with
                // their partner's value
                val += x[lane + i];
      
            // synchronize all threads in group
            g.sync();
      
        }

    // master thread in group returns result, and others return -1.
    if (g.thread_rank()==0)
        return val; 
    else
        return -1;
}


/**
 * CUDA kernel device code
 * 
 * Creates cooperative groups and performs reductions
 */
__global__ void cgkernel(){


    // threadBlockGroup includes all threads in the block
    thread_block threadBlockGroup = this_thread_block();
    int threadBlockGroupSize=threadBlockGroup.size();

    // workspace array in shared memory required for reduction
    extern __shared__ int workspace[];

    int input, output, expectedOutput;

    // input to reduction, for each thread, is its' rank in the group
    input=threadBlockGroup.thread_rank();

    // expected output from analytical formula (n-1)(n)/2
    // (noting that indexing starts at 0 rather than 1)
    expectedOutput=(threadBlockGroupSize-1)*threadBlockGroupSize/2;

    // perform reduction
    output=sumReduction(threadBlockGroup, workspace, input);

    // master thread in group prints out result
    if(threadBlockGroup.thread_rank()==0){
    
        printf(" Sum of all ranks 0..%d in threadBlockGroup is %d (expected %d)\n\n",
               threadBlockGroup.size()-1,output,
               expectedOutput);
    
        printf(" Now creating %d groups, each of size 16 threads:\n\n",
               threadBlockGroup.size()/16);
    
    }

    threadBlockGroup.sync();
  
    // each tiledPartition16 group includes 16 threads
    thread_block_tile<16> tiledPartition16 = tiled_partition<16>(threadBlockGroup);

    // This offset allows each group to have its own unique area in the workspace array
    int workspaceOffset=threadBlockGroup.thread_rank()-tiledPartition16.thread_rank();

    // input to reduction, for each thread, is its' rank in the group
    input=tiledPartition16.thread_rank();

    // expected output from analytical formula (n-1)(n)/2
    // (noting that indexing starts at 0 rather than 1)
    expectedOutput=15*16/2;
  
    // Perform reduction
    output=sumReduction(tiledPartition16, workspace+workspaceOffset, input);

    // each master thread prints out result
    if(tiledPartition16.thread_rank()==0)
        printf("   Sum of all ranks 0..15 in this tiledPartition16 group is %d (expected %d)\n",output,expectedOutput);

    return;
  
}


/**
 * Host main routine
 */
int main(){

    // Error code to check return values for CUDA calls
    cudaError_t err;

    //Launch the kernel

    int blocksPerGrid=1;
    int threadsPerBlock=64;

    printf("\nLaunching a single block with %d threads...\n\n",threadsPerBlock);

    // we use the optional third argument to specify the size
    // of shared memory required in the kernel
    cgkernel <<<blocksPerGrid,threadsPerBlock,threadsPerBlock*sizeof(int)>>> (); 
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    printf("\n...Done.\n\n");

    return 0;
}
