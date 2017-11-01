__global__ void simple_kernel(const int *pIn, int *pOut, int a)
{
    __shared__ int sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    pOut[tid] = sData[threadIdx.x]*a + tid;;
}

__global__ void simple_kernel(const int2 *pIn, int *pOut, int a)
{
    __shared__ int2 sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    pOut[tid] = (sData[threadIdx.x].x + sData[threadIdx.x].y)*a + tid;;
}

__global__ void simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a)
{
    __shared__ int sData1[THREAD_N];
    __shared__ int sData2[THREAD_N];
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    sData1[threadIdx.x] = pIn1[tid];
    sData2[threadIdx.x] = pIn2[tid];
    __syncthreads();

    pOut[tid] = (sData1[threadIdx.x] + sData2[threadIdx.x])*a + tid;
}
