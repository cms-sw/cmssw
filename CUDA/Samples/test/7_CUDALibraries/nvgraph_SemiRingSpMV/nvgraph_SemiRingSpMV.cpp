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
#include "helper_cuda.h"
#include "nvgraph.h"

/*
----------------------------------
Initially :

V = 5
E = 10

y = (0.0, 0.0, 0.0, 0.0, 0.0)^T
alpha = 1.0 
beta = 0.0
         G               x
1.0 4.0 0.0 0.0 0.0     1.1
0.0 2.0 3.0 0.0 0.0     2.2
5.0 0.0 0.0 7.0 8.0     3.3
0.0 0.0 9.0 0.0 6.0     4.4
0.0 0.0 1.5 0.0 0.0     5.5

Destination oriented representation :
source_offsets {0, 2, 4, 7, 9, 10}
destination_indices {0, 1, 1, 2, 0, 3, 4, 2, 4, 2}
W0 = {1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 6.0, 1.5}

----------------------------------

Operation :
y = alaph SR_time G SR_time x SR_plus y
----------------------------------

Expected output : y = (9.9, 14.3, 80.3, 62.7, 4.95)^T



 NVGRAPH's semi-ring SPMV

 Performs a semi-ring operation using a graph and input vector
  On input: x contains a dense vector of values
  On output: y =  alpha *SPMV[sr]( x ) + beta *y
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

    const size_t  n = 5, nnz = 10, vertex_numsets = 2, edge_numsets = 1;
    const float alpha = 1.0, beta = 0.0;
    const void *alpha_p = (const void *) &alpha, *beta_p = (const void *) &beta;
    int i, *source_offsets_h, *destination_indices_h;
    float *weights_h, *x_h, *y_h;
    void** vertex_dim;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    
    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;

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

    // Init host data
    source_offsets_h = (int*) malloc((n+1)*sizeof(int));
    destination_indices_h = (int*) malloc(nnz*sizeof(int));
    weights_h = (float*)malloc(nnz*sizeof(float));
    x_h = (float*)malloc(n*sizeof(float));
    y_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

    vertex_dim[0]= (void*)x_h; vertex_dim[1]= (void*)y_h;
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F;

    weights_h [0] = 1.0f;
    weights_h [1] = 4.0f;
    weights_h [2] = 2.0f;
    weights_h [3] = 3.0f;
    weights_h [4] = 5.0f;
    weights_h [5] = 7.0f;
    weights_h [6] = 8.0f;
    weights_h [7] = 9.0f;
    weights_h [8] = 6.0f;
    weights_h [9] = 1.5f;

    source_offsets_h [0] = 0;
    source_offsets_h [1] = 2;
    source_offsets_h [2] = 4;
    source_offsets_h [3] = 7;
    source_offsets_h [4] = 9;
    source_offsets_h [5] = 10;

    destination_indices_h [0] = 0;
    destination_indices_h [1] = 1;
    destination_indices_h [2] = 1;
    destination_indices_h [3] = 2;
    destination_indices_h [4] = 0;
    destination_indices_h [5] = 3;
    destination_indices_h [6] = 4;
    destination_indices_h [7] = 2;
    destination_indices_h [8] = 4;
    destination_indices_h [9] = 2;

    x_h[0] = 1.1f;
    x_h[1] = 2.2f;
    x_h[2] = 3.3f;
    x_h[3] = 4.4f;
    x_h[4] = 5.5f;

    y_h[0] = 0.0f;
    y_h[1] = 0.0f;
    y_h[2] = 0.0f;
    y_h[3] = 0.0f;
    y_h[4] = 0.0f;

    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));

    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets_h;
    CSR_input->destination_indices = destination_indices_h;

    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    for (i = 0; i < vertex_numsets; ++i)
       check_status(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    // Solve
    check_status(nvgraphSrSpmv(handle, graph, 0, alpha_p, 0, beta_p, 1, NVGRAPH_PLUS_TIMES_SR));

    //Get and print result
    check_status(nvgraphGetVertexData(handle, graph, (void*)y_h, 1));
    printf("y_h\n");
    for (i = 0; i<n; i++)  printf("%f\n",y_h[i]); printf("\n");
    printf("\nDone!\n");

    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));

    free(source_offsets_h);
    free(destination_indices_h);
    free(weights_h);
    free(x_h);
    free(y_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSR_input);

    return EXIT_SUCCESS;
}
