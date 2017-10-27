#include "eglstrm_common.h"
#include <cuda.h>

int     parseCmdLine(int argc, char *argv[], TestArgs *args);
void    printUsage(void);
int     NUMTRIALS = 10;
int     profileAPIs = 0;

bool verbose = 0;
bool isCrossDevice = 0;

// Parse the command line options. Returns FAILURE on a parse error, SUCCESS
// otherwise.
int parseCmdLine(int argc, char *argv[], TestArgs *args)
{
    int i;

    for(i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) {
            printUsage();
            exit(0);
        } else if (strcmp(argv[i], "-n") == 0) {
            ++i;
            if (sscanf(argv[i], "%d", &NUMTRIALS) != 1 || NUMTRIALS <= 0) {
                printf("Invalid trial count: %s should be > 0\n", argv[i]);
                return -1 ;
            }
        } else if (strcmp(argv[i], "-profile") == 0) {
            profileAPIs = 1;
        } else if (strcmp(argv[i], "-crossdev") == 0) {
            isCrossDevice = 1;
        } else if (strcmp(argv[i], "-dgpu") == 0) {
            isCrossDevice = 1;
        } else if (strcmp(argv[i], "-width") == 0) {
            ++i;
            if (sscanf(argv[i], "%d", &WIDTH) != 1 || (WIDTH <= 0)) {
                printf("Width should be greater than 0\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-height") == 0) {
            ++i;
            if (sscanf(argv[i], "%d", &HEIGHT) != 1 || (HEIGHT<= 0)) {
                printf("Width should be greater than 0\n");
                return -1;
            }
        } else if (0 == strcmp(&argv[i][1], "proctype")) {
            ++i;
            if (!strcasecmp(argv[i], "prod")) {
                        args->isProducer = 1;
            } else if (!strcasecmp(argv[i], "cons")) {
                args->isProducer = 0;
            }
            else {
                printf("%s: Bad Process Type: %s\n",__func__, argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        }
        else {
            printf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (isCrossDevice)
    {
        int deviceCount = 0;
        CUresult error_id = cuInit(0);

        if (error_id != CUDA_SUCCESS)
        {
            printf("cuInit(0) returned %d\n", error_id);
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }
    
        error_id = cuDeviceGetCount(&deviceCount);
    
        if (error_id != CUDA_SUCCESS)
        {
            printf("cuDeviceGetCount returned %d\n", (int)error_id);
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        int iGPUexists = 0;
        CUdevice dev;
        for (dev = 0; dev < deviceCount; ++dev)
        {
            int integrated = 0;
            CUresult error_result = cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);

            if (error_result != CUDA_SUCCESS)
            {
                printf("cuDeviceGetAttribute returned error : %d\n", (int)error_result);
                exit(EXIT_FAILURE);
            }

            if (integrated)
            {
                iGPUexists = 1;
            }
        }

        if (!iGPUexists)
        {
            printf("No Integrated GPU found in the system.\n");
            printf("-dgpu or -crossdev option is only supported on systems with an Integrated GPU and a Discrete GPU\n");
            printf("Waiving the execution\n");
            exit(EXIT_SUCCESS);
        }
    }

    if (!args->isProducer) {
        /* Cross-process creation of producer */
        char argsProducer[1024];
        char str[256];

        strcpy(argsProducer,"./EGLStream_CUDA_CrossGPU -proctype prod ");

        if (isCrossDevice)
        {
            sprintf(str,"-crossdev ");
            strcat(argsProducer,str);
        }

        if (verbose)
        {
            sprintf(str,"-v ");
            strcat(argsProducer,str);
        }

        /*Make the process run in bg*/
        strcat(argsProducer,"& ");

        printf("\n%s: Crossproc Producer command: %s \n", __func__, argsProducer);

        /*Create crossproc Producer*/
        system(argsProducer);

        /*Enable crossproc Consumer in the same process */
        args->isProducer = 0;
    }

    return 0;
}


void printUsage(void)
{
    printf("Usage:\n");
    printf("  -h           Print this help message\n");
    printf("  -n n         Exit after running n trials. Set to 10 by default\n");
    printf("  -profile     Profile time taken by ReleaseAPI. Not set by default\n");
    printf("  -crossdev    Run with producer on idgpu and consumer on dgpu\n");
    printf("  -dgpu        (same as -crossdev, deprecated)\n");
    printf("  -v           verbose output\n");
}
