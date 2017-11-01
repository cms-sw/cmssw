/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
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

/* Spectral Clustering Sample


Social network example: Zachary Karate Club 
W. Zachary, “An information flow model for conflict and fission in small groups,” Journal of Anthropological Research, vol. 33, pp. 452–473, 1977
https://en.wikipedia.org/wiki/Zachary's_karate_club

The vertices represent the 34 members in a Karate club. 
An edge between two vertices indicates that the two members spent significant time together outside normal club meetings. 
While Zachary was collecting his data, there was a dispute in the Karate club, and it split into two factions: 
one led by “Mr. Hi” (cluster #1), and one led by “John A” (cluster #0).
Reference factions {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

--------------------------------------------------------------------
V = 34
E = 78 bidirectional, 156 directed edges

Bidirectional edges list:
[2 1] [3 1] [3 2] [4 1] [4 2] [4 3] [5 1] [6 1] [7 1] [7 5] [7 6] [8 1] [8 2] [8 3] [8 4] [9 1] [9 3] [10 3] [11 1] [11 5] [11 6] [12 1] [13 1] [13 4] [14 1] [14 2] [14 3] [14 4] [17 6] [17 7] 
[18 1] [18 2] [20 1] [20 2] [22 1] [22 2] [26 24] [26 25] [28 3] [28 24] [28 25] [29 3] [30 24] [30 27] [31 2] [31 9] [32 1] [32 25] [32 26] [32 29] [33 3] [33 9] [33 15] [33 16] 
[33 19] [33 21] [33 23] [33 24] [33 30] [33 31] [33 32] [34 9] [34 10] [34 14] [34 15] [34 16] [34 19] [34 20] [34 21] [34 23] [34 24] [34 27] [34 28] [34 29] [34 30] [34 31] 
[34 32] [34 33]

CSR representation (directed):
csrRowPtrA_h {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156}
csrColIndA_h {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 6, 10, 16, 0, 
4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 
24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 
26, 27, 28, 29, 30, 31, 32}
csrValA_h {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

--------------------------------------------------------------------

Operation: Spectral Clustering looking for 2 clusters with 2 eigenpairs, default parameters in modularity maximization

--------------------------------------------------------------------

Expected output: 
1) Assignments in 2 clusters with modularity score between 0.0 and 1.0
2) This sample prints the hit rate of nvgraph's spectral approximation compared to the real factions of Mr. Hi and John A.

*/

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %s\n",nvgraphStatusGetString(status));
        exit(0);
    }
}

int main(int argc, char **argv)
{
    // Hard-coded Zachary Karate Club network input
    int csrRowPtrA_input [] = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
        139, 156};
    int csrColIndA_input [] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
        6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
        25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
        18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
    float csrValA_input [] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int ref_clustering [] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int *csrRowPtrA_h = &csrRowPtrA_input[0];
    int *csrColIndA_h = &csrColIndA_input[0];
    float *csrValA_h = &csrValA_input[0];

    // Variables
    const size_t  n = 34, nnz = 156, n_ev = 2, edge_numsets = 1;
    int i, weight_index = 0, *clustering_h, hits=0;
    float *eigVals_h, *eigVecs_h, modularity_score = 0.0;  
    
    // Nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    nvgraphCSRTopology32I_st CSR_input = {n, nnz, csrRowPtrA_h, csrColIndA_h};
    
    // Allocate host data for nvgraphSpectralClustering output
    clustering_h = (int*)malloc(n*sizeof(int));
    eigVals_h = (float*)malloc(n_ev*sizeof(float));
    eigVecs_h = (float*)malloc(n_ev*n*sizeof(float));
    
    // Spectral clustering parameters
    struct SpectralClusteringParameter clustering_params;
    clustering_params.n_clusters = n_ev; 
    clustering_params.n_eig_vects = n_ev; 
    clustering_params.algorithm = NVGRAPH_MODULARITY_MAXIMIZATION ; 
    clustering_params.evs_tolerance = 0.0f ; //automatically selects default value
    clustering_params.evs_max_iter = 0; //automatically selects default value
    clustering_params.kmean_tolerance = 0.0f; //automatically selects default value
    clustering_params.kmean_max_iter = 0; //automatically selects default value

    findCudaDevice(argc, (const char **)argv);

    // Starting nvgraph
    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)&CSR_input, NVGRAPH_CSR_32));
    check_status(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT ));
    check_status(nvgraphSetEdgeData(handle, graph, (void*) csrValA_h, 0 ));   
    
    //Solve spectral clustering with modularity maximization algorithm
    //looking for 2 clusters with 2 eigenpairs, default parameters
    check_status(nvgraphSpectralClustering(handle, graph, weight_index, &clustering_params, clustering_h, eigVals_h, eigVecs_h)); 

    //Analyze quality (modualrity)
    check_status(nvgraphAnalyzeClustering(handle, graph, weight_index, clustering_params.n_clusters, clustering_h, NVGRAPH_MODULARITY, &modularity_score));  
    printf("Modularity_score: %f\n", modularity_score);
    
    //Number of hits compared to the real factions
    for (i = 0; i < (int)n; i++)
        if (clustering_h[i] == ref_clustering[i])
            hits++;
     printf("Hit rate : %f%% (%d hits)\n", (hits*100.0)/n, hits);

    //Exit
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle)); 
    free(clustering_h);
    free(eigVals_h);
    free(eigVecs_h);
    printf("Done!\n");

    return EXIT_SUCCESS;
}

