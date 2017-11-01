extern "C" __global__ void sequence_gpu(int *d_ptr, int length) 
{ 
    int elemID = blockIdx.x * blockDim.x + threadIdx.x; 

    if (elemID < length)
    {
        unsigned int laneid;

        //This command gets the lane ID within the current warp
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

        d_ptr[elemID] = laneid;
    }
}

