/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates Inter Process Communication
 *  features new to SDK 4.1 and uses one process per GPU for computation.
 * Note: Multiple processes per single device are possible but not recommended.
 *       In such cases, one should use IPC events for hardware synchronization.
 */

// Includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime includes
#include <cuda_runtime_api.h>

// CUDA utilities and system includes
#include <helper_cuda.h>

int   *pArgc = NULL;
char **pArgv = NULL;

#define MAX_DEVICES          8
#define PROCESSES_PER_DEVICE 1
#define DATA_BUF_SIZE        4096

#ifdef __linux
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <linux/version.h>

typedef struct ipcCUDA_st
{
    int device;
    pid_t pid;
    cudaIpcEventHandle_t eventHandle;
    cudaIpcMemHandle_t memHandle;
} ipcCUDA_t;

typedef struct ipcDevices_st
{
    int count;
    int ordinals[MAX_DEVICES];
} ipcDevices_t;

typedef struct ipcBarrier_st
{
    int count;
    bool sense;
    bool allExit;
} ipcBarrier_t;

ipcBarrier_t *g_barrier = NULL;
bool          g_procSense;
int           g_processCount;

void procBarrier()
{
    int newCount = __sync_add_and_fetch(&g_barrier->count, 1);

    if (newCount == g_processCount)
    {
        g_barrier->count = 0;
        g_barrier->sense = !g_procSense;
    }
    else
    {
        while (g_barrier->sense == g_procSense)
        {
            if (!g_barrier->allExit)
            {
                sched_yield();
            }
            else
            {
                exit(EXIT_FAILURE);
            }
        }
    }

    g_procSense = !g_procSense;
}

// CUDA Kernel
__global__ void simpleKernel(int *dst, int *src, int num)
{
    // Dummy kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] / num;
}

void getDeviceCount(ipcDevices_t *devices)
{
    // We can't initialize CUDA before fork() so we need to spawn a new process

    pid_t pid = fork();

    if (0 == pid)
    {
        int i;
        int count, uvaCount = 0;
        int uvaOrdinals[MAX_DEVICES];
        printf("\nChecking for multiple GPUs...\n");
        checkCudaErrors(cudaGetDeviceCount(&count));
        printf("CUDA-capable device count: %i\n", count);

        printf("\nSearching for UVA capable devices...\n");

        for (i = 0; i < count; i++)
        {
            cudaDeviceProp prop;
            checkCudaErrors(cudaGetDeviceProperties(&prop, i));

            if (prop.unifiedAddressing)
            {
                uvaOrdinals[uvaCount] = i;
                printf("> GPU%d = \"%15s\" IS capable of UVA\n", i, prop.name);
                uvaCount += 1;
            }

            if (prop.computeMode != cudaComputeModeDefault)
            {
                printf("> GPU device must be in Compute Mode Default to run\n");
                printf("> Please use nvidia-smi to change the Compute Mode to Default\n");
                exit(EXIT_SUCCESS);
            }
        }

        devices->ordinals[0] = uvaOrdinals[0];

        if (uvaCount < 2)
        {
            devices->count = uvaCount;
            exit(EXIT_SUCCESS);
        }

        // Check possibility for peer accesses, relevant to our tests
        printf("\nChecking GPU(s) for support of peer to peer memory access...\n");
        devices->count = 1;
        int canAccessPeer_0i, canAccessPeer_i0;

        for (i = 1; i < uvaCount; i++)
        {
            checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer_0i, uvaOrdinals[0], uvaOrdinals[i]));
            checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer_i0, uvaOrdinals[i], uvaOrdinals[0]));

            if (canAccessPeer_0i*canAccessPeer_i0)
            {
                devices->ordinals[devices->count] = uvaOrdinals[i];
                printf("> Two-way peer access between GPU%d and GPU%d: YES\n", devices->ordinals[0], devices->ordinals[devices->count]);
                devices->count += 1;
            }
        }

        exit(EXIT_SUCCESS);
    }
    else
    {
        int status;
        waitpid(pid, &status, 0);
        assert(!status);
    }
}

inline bool IsAppBuiltAs64()
{
    return sizeof(void*) == 8;
}

void runTestMultiKernel(ipcCUDA_t *s_mem, int index)
{
    /*
     * a) Process 0 loads a reference buffer into GPU0 memory
     * b) Other processes launch a kernel on the GPU0 memory using P2P
     * c) Process 0 checks the resulting buffer
     */

    // memory buffer in gpu
    int *d_ptr;

    // reference buffer in host memory  (do in all processes for rand() consistency)
    int h_refData[DATA_BUF_SIZE];

    for (int i = 0; i < DATA_BUF_SIZE; i++)
    {
        h_refData[i] = rand();
    }

    checkCudaErrors(cudaSetDevice(s_mem[index].device));

    if (index == 0)
    {
        printf("\nLaunching kernels...\n");
        // host memory buffer for checking results
        int h_results[DATA_BUF_SIZE * MAX_DEVICES * PROCESSES_PER_DEVICE];

        cudaEvent_t event[MAX_DEVICES * PROCESSES_PER_DEVICE];
        checkCudaErrors(cudaMalloc((void **) &d_ptr, DATA_BUF_SIZE * g_processCount * sizeof(int)));
        checkCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &s_mem[0].memHandle, (void *) d_ptr));
        checkCudaErrors(cudaMemcpy((void *) d_ptr, (void *) h_refData, DATA_BUF_SIZE * sizeof(int), cudaMemcpyHostToDevice));

        // b.1: wait until all event handles are created in other processes
        procBarrier();

        for (int i = 1; i < g_processCount; i++)
        {
            checkCudaErrors(cudaIpcOpenEventHandle(&event[i], s_mem[i].eventHandle));
        }

        // b.2: wait until all kernels launched and events recorded
        procBarrier();

        for (int i = 1; i < g_processCount; i++)
        {
            checkCudaErrors(cudaEventSynchronize(event[i]));
        }

        // b.3
        procBarrier();

        checkCudaErrors(cudaMemcpy(h_results, d_ptr + DATA_BUF_SIZE,
                                   DATA_BUF_SIZE * (g_processCount - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_ptr));
        printf("Checking test results...\n");

        for (int n = 1; n < g_processCount; n++)
        {
            for (int i = 0; i < DATA_BUF_SIZE; i++)
            {
                if (h_refData[i]/(n + 1) != h_results[(n-1) * DATA_BUF_SIZE + i])
                {
                    fprintf(stderr, "Data check error at index %d in process %d!: %i,    %i\n",i,
                            n, h_refData[i], h_results[(n-1) * DATA_BUF_SIZE + i]);
                    g_barrier->allExit = true;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    else
    {
        cudaEvent_t event;
        checkCudaErrors(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));
        checkCudaErrors(cudaIpcGetEventHandle((cudaIpcEventHandle_t *) &s_mem[index].eventHandle, event));

        // b.1: wait until proc 0 initializes device memory
        procBarrier();

        checkCudaErrors(cudaIpcOpenMemHandle((void **) &d_ptr, s_mem[0].memHandle,
                                             cudaIpcMemLazyEnablePeerAccess));
        printf("> Process %3d: Run kernel on GPU%d, taking source data from and writing results to process %d, GPU%d...\n",
               index, s_mem[index].device, 0, s_mem[0].device);
        const dim3 threads(512, 1);
        const dim3 blocks(DATA_BUF_SIZE / threads.x, 1);
        simpleKernel<<<blocks, threads>>> (d_ptr + index *DATA_BUF_SIZE, d_ptr, index + 1);
        checkCudaErrors(cudaEventRecord(event));

        // b.2
        procBarrier();

        checkCudaErrors(cudaIpcCloseMemHandle(d_ptr));

        // b.3: wait till all the events are used up by proc g_processCount - 1
        procBarrier();

        checkCudaErrors(cudaEventDestroy(event));
    }

    cudaDeviceReset();

}
#endif

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;


#if CUDART_VERSION >= 4010 && defined(__linux)

    if (!IsAppBuiltAs64())
    {
        printf("%s is only supported on 64-bit Linux OS and the application must be built as a 64-bit target. Test is being waived.\n", argv[0]);
        exit(EXIT_WAIVED);
    }

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18)
    printf("%s is only supported with Linux OS kernel version 2.6.18 and higher. Test is being waived.\n", argv[0]);
    exit(EXIT_WAIVED);
#endif

    ipcDevices_t *s_devices = (ipcDevices_t *) mmap(NULL, sizeof(*s_devices),
                                                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != s_devices);

    // We can't initialize CUDA before fork() so we need to spawn a new process
    getDeviceCount(s_devices);

    if (s_devices->count < 1)
    {
        printf("One or more (SM 2.0) class GPUs are required for %s.\n", argv[0]);
        printf("Waiving test.\n");
        exit(EXIT_SUCCESS);
    }

    // initialize our process and barrier data
    // if there is more than one device, 1 process per device
    if (s_devices->count > 1)
    {
        g_processCount = PROCESSES_PER_DEVICE * s_devices->count;
    }
    else
    {
        g_processCount = 2; // two processes per single device
    }

    g_barrier = (ipcBarrier_t *) mmap(NULL, sizeof(*g_barrier),
                                      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != g_barrier);
    memset((void *) g_barrier, 0, sizeof(*g_barrier));
    // set local barrier sense flag
    g_procSense = 0;

    // shared memory for CUDA memory an event handlers
    ipcCUDA_t *s_mem = (ipcCUDA_t *) mmap(NULL, g_processCount * sizeof(*s_mem),
                                          PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != s_mem);

    // initialize shared memory
    memset((void *) s_mem, 0, g_processCount * sizeof(*s_mem));

    printf("\nSpawning processes and assigning GPUs...\n");

    // index = 0,.., g_processCount - 1
    int index = 0;

    // spawn "g_processCount - 1" additional processes
    for (int i = 1; i < g_processCount; i++)
    {
        int pid = fork();

        if (!pid)
        {
            index = i;
            break;
        }
        else
        {
            s_mem[i].pid = pid;
        }
    }

    // distribute UVA capable devices among processes (1 device per PROCESSES_PER_DEVICE processes)
    // if there is only one device, have 1 extra process
    if (s_devices->count > 1)
    {
        s_mem[index].device = s_devices->ordinals[ index / PROCESSES_PER_DEVICE ];
    }
    else
    {
        s_mem[0].device = s_mem[1].device = s_devices->ordinals[ 0 ];
    }

    printf("> Process %3d -> GPU%d\n", index, s_mem[index].device);

    // launch our test
    runTestMultiKernel(s_mem, index);

    // Cleanup and shutdown
    if (index == 0)
    {
        // wait for processes to complete
        for (int i = 1; i < g_processCount; i++)
        {
            int status;
            waitpid(s_mem[i].pid, &status, 0);
            assert(WIFEXITED(status));
        }

        printf("\nShutting down...\n");

        exit(EXIT_SUCCESS);
    }

#else // Using CUDA 4.0 and older or non Linux OS
    printf("simpleIPC requires CUDA 4.1 and Linux to build and run, waiving testing\n\n");
    exit(EXIT_WAIVED);
#endif
}
