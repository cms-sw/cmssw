/*
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

/* PageRank
 *  Find PageRank for a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
 *  This is equivalent to an eigenvalue problem where we want the eigenvector corresponding to the maximum eigenvalue.
 *  By construction, the maximum eigenvalue is 1.
 *  The eigenvalue problem is solved with the power method.

Initially :
V = 6 
E = 10

Edges       W
0 -> 1    0.50
0 -> 2    0.50
2 -> 0    0.33
2 -> 1    0.33
2 -> 4    0.33
3 -> 4    0.50
3 -> 5    0.50
4 -> 3    0.50
4 -> 5    0.50
5 -> 3    1.00

bookmark (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)^T note: 1.0 if i is a dangling node, 0.0 otherwise

Source oriented representation (CSC):
destination_offsets {0, 1, 3, 4, 6, 8, 10}
source_indices {2, 0, 2, 0, 4, 5, 2, 3, 3, 4}
W0 = {0.33, 0.50, 0.33, 0.50, 0.50, 1.00, 0.33, 0.50, 0.50, 1.00}

----------------------------------

Operation : Pagerank with various damping factor 
----------------------------------

Expected output for alpha= 0.9 (result stored in pr_2) : (0.037210, 0.053960, 0.041510, 0.37510, 0.206000, 0.28620)^T 
From "Google's PageRank and Beyond: The Science of Search Engine Rankings" Amy N. Langville & Carl D. Meyer
*/

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

int main(int argc, char **argv)
{
    const size_t  n = 6, nnz = 10, vertex_numsets = 3, edge_numsets = 1;
    const float alpha1 = 0.85, alpha2 = 0.90;
    const void *alpha1_p = (const void *) &alpha1, *alpha2_p = (const void *) &alpha2;
    int i, *destination_offsets_h, *source_indices_h;
    float *weights_h, *bookmark_h, *pr_1,*pr_2;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int cuda_device = 0;
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    if (deviceProp.major < 3)
    {
        printf("> nvGraph requires device SM 3.0+\n");
        printf("> Waiving.\n");
        exit(EXIT_WAIVED);
    }


    // Allocate host data
    destination_offsets_h = (int*) malloc((n+1)*sizeof(int));
    source_indices_h = (int*) malloc(nnz*sizeof(int));
    weights_h = (float*)malloc(nnz*sizeof(float));
    bookmark_h = (float*)malloc(n*sizeof(float));
    pr_1 = (float*)malloc(n*sizeof(float));
    pr_2 = (float*)malloc(n*sizeof(float));
    vertex_dim = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    // Initialize host data
    vertex_dim[0] = (void*)bookmark_h; vertex_dim[1]= (void*)pr_1, vertex_dim[2]= (void*)pr_2;
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F, vertex_dimT[2]= CUDA_R_32F;
    
    weights_h [0] = 0.333333f;
    weights_h [1] = 0.500000f;
    weights_h [2] = 0.333333f;
    weights_h [3] = 0.500000f;
    weights_h [4] = 0.500000f;
    weights_h [5] = 1.000000f;
    weights_h [6] = 0.333333f;
    weights_h [7] = 0.500000f;
    weights_h [8] = 0.500000f;
    weights_h [9] = 0.500000f;

    destination_offsets_h [0] = 0;
    destination_offsets_h [1] = 1;
    destination_offsets_h [2] = 3;
    destination_offsets_h [3] = 4;
    destination_offsets_h [4] = 6;
    destination_offsets_h [5] = 8;
    destination_offsets_h [6] = 10;

    source_indices_h [0] = 2;
    source_indices_h [1] = 0;
    source_indices_h [2] = 2;
    source_indices_h [3] = 0;
    source_indices_h [4] = 4;
    source_indices_h [5] = 5;
    source_indices_h [6] = 2;
    source_indices_h [7] = 3;
    source_indices_h [8] = 3;
    source_indices_h [9] = 4;

    bookmark_h[0] = 0.0f;
    bookmark_h[1] = 1.0f;
    bookmark_h[2] = 0.0f;
    bookmark_h[3] = 0.0f;
    bookmark_h[4] = 0.0f;
    bookmark_h[5] = 0.0f;

    // Starting nvgraph
    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    
    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    for (i = 0; i < 2; ++i)
        check_status(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    
    // First run with default values
    check_status(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));
    
    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
    printf("pr_1, alpha = 0.85\n"); for (i = 0; i<n; i++)  printf("%f\n",pr_1[i]); printf("\n");

    // Second run with different damping factor and an initial guess
    for (i = 0; i<n; i++)  
        pr_2[i] =pr_1[i]; 
   
    nvgraphSetVertexData(handle, graph, vertex_dim[2], 2);
    check_status(nvgraphPagerank(handle, graph, 0, alpha2_p, 0, 1, 2, 0.0f, 0));

    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[2], 2));
    printf("pr_2, alpha = 0.90\n"); for (i = 0; i<n; i++)  printf("%f\n",pr_2[i]); printf("\n");

    //Clean 
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(bookmark_h);
    free(pr_1);
    free(pr_2);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    printf("\nDone!\n");
    return EXIT_SUCCESS;
}
