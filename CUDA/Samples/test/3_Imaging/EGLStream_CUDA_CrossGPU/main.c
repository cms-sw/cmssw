/*
 * testmain.c
 *
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuda_consumer.h"
#include "cuda_producer.h"
#include "eglstrm_common.h"
#include "cudaEGL.h"
#include "helper.h"
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

bool signal_stop = 0;
extern bool verbose;

static void
sig_handler(int sig)
{
    signal_stop = 1;
    printf("Signal: %d\n", sig);
}

int WIDTH = 8192, HEIGHT = 8192;
int
main(int argc, char **argv)
{
    TestArgs args;
    CUresult curesult = CUDA_SUCCESS;
    unsigned int j = 0;
    cudaError_t err = cudaSuccess;
    double curTime;
    EGLNativeFileDescriptorKHR fileDescriptor = EGL_NO_FILE_DESCRIPTOR_KHR;
    struct timespec start, end;
    CUeglFrame cudaEgl1, cudaEgl2;
    pid_t childPID;

    if (parseCmdLine(argc, argv, &args)  < 0) {
        printUsage();
        curesult = CUDA_ERROR_UNKNOWN;
        goto DoneCons;
    }

    printf("Width : %u, height: %u and iterations: %u\n", WIDTH, HEIGHT, NUMTRIALS);

    if (!args.isProducer) // Consumer code
    {
        test_cuda_consumer_s cudaConsumer;
        memset(&cudaConsumer, 0, sizeof(test_cuda_consumer_s));
        cudaConsumer.profileAPI = profileAPIs;
        int consumerStatus = 0;
        // Hook up Ctrl-C handler
        signal(SIGINT, sig_handler);

        curesult = cudaDeviceCreateConsumer(&cudaConsumer);
        if (curesult != CUDA_SUCCESS) {
            consumerStatus = -1;
            goto DoneCons;
        }

        cuCtxPushCurrent(cudaConsumer.context);

        if (!eglSetupExtensions(isCrossDevice)) {
            printf("SetupExtentions failed \n");
            curesult = CUDA_ERROR_UNKNOWN;
            consumerStatus = -1;
            goto DoneCons;
        }
        if (!EGLStreamInit(isCrossDevice, !args.isProducer, EGL_NO_FILE_DESCRIPTOR_KHR)) {
            printf("EGLStream Init failed.\n");
            curesult = CUDA_ERROR_UNKNOWN;
            consumerStatus = -1;
            goto DoneCons;
        }

        args.charCnt = WIDTH * HEIGHT * 4;

        curesult = cuda_consumer_init(&cudaConsumer, &args);
        if (curesult != CUDA_SUCCESS) {
            printf("Cuda Consumer: Init failed, status: %d\n", curesult);
            consumerStatus = -1;
            goto DoneCons;
        }

        cuCtxPopCurrent(&cudaConsumer.context);

        int send_fd = -1;
        send_fd = UnixSocketConnect(SOCK_PATH);
        if (-1 == send_fd){
           printf("%s: Cuda Consumer cannot create socket %s\n", __func__, SOCK_PATH);
            consumerStatus = -1;
           goto DoneCons;
        }

        cuCtxPushCurrent(cudaConsumer.context);
        cudaConsumer.eglStream = g_consumerEglStream;
        cudaConsumer.eglDisplay = g_consumerEglDisplay;

        //Send the EGL stream FD to producer 
        fileDescriptor = eglGetStreamFileDescriptorKHR(cudaConsumer.eglDisplay, cudaConsumer.eglStream);
        if (EGL_NO_FILE_DESCRIPTOR_KHR == fileDescriptor) {
            printf("%s: Cuda Consumer could not get EGL file descriptor.\n",__func__);
            eglDestroyStreamKHR(cudaConsumer.eglDisplay, cudaConsumer.eglStream);
            consumerStatus = -1;
            goto DoneCons;
        }

        if (verbose)
            printf("%s: Cuda Consumer EGL stream FD obtained : %d.\n",__func__, fileDescriptor);

        int res = -1;
        res = EGLStreamSendfd(send_fd, fileDescriptor);
        if (-1 == res) {
            printf("%s: Cuda Consumer could not send EGL file descriptor.\n", __func__);
            consumerStatus = -1;
            close(fileDescriptor);
        }

        if (CUDA_SUCCESS != (curesult = cuEGLStreamConsumerConnect(&(cudaConsumer.cudaConn), cudaConsumer.eglStream))) {
            printf("FAILED Connect CUDA consumer with error %d\n", curesult);
            consumerStatus = -1;
            goto DoneCons;
        }

        j=0;
        for (j = 0;  j < NUMTRIALS; j++)
        {
            curesult = cudaConsumerAcquireFrame(&cudaConsumer, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Consumer Test failed for frame = %d\n", j+1);
                consumerStatus = -1;
                goto DoneCons;
            }
            curesult = cudaConsumerReleaseFrame(&cudaConsumer, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Consumer Test failed for frame = %d\n", j+1);
                consumerStatus = -1;
                goto DoneCons;
            }

            curesult = cudaConsumerAcquireFrame(&cudaConsumer, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Consumer Test failed for frame = %d\n", j+1);
                consumerStatus = -1;
                goto DoneCons;
            }
            curesult = cudaConsumerReleaseFrame(&cudaConsumer, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Consumer Test failed for frame = %d\n", j+1);
                consumerStatus = -1;
                goto DoneCons;
            }
        }
        cuCtxSynchronize();
        close(fileDescriptor);
        err =  cudaGetValueMismatch();
        if (err != cudaSuccess) {
            printf("Consumer: App failed with value mismatch\n");
            curesult = CUDA_ERROR_UNKNOWN;
            consumerStatus = -1;
            goto DoneCons;
        }

        EGLint streamState = 0;
        if (!eglQueryStreamKHR(
                    cudaConsumer.eglDisplay,
                    cudaConsumer.eglStream,
                    EGL_STREAM_STATE_KHR,
                    &streamState)) {
            printf("Main, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
            curesult = CUDA_ERROR_UNKNOWN;
            consumerStatus = -1;
            goto DoneCons;
        }

        if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
            if (CUDA_SUCCESS != (curesult = cuda_consumer_Deinit(&cudaConsumer))) {
                printf("Consumer Disconnect FAILED.\n");
                consumerStatus = -1;
                goto DoneCons;
            }
        }

DoneCons:
        EGLStreamFini();
        //get the final status from producer, combine and print
        int producerStatus = -1;
        if (-1 == recv(send_fd,(void*)&producerStatus, sizeof(int), 0)) {
            printf("%s: Cuda Consumer could not receive status from producer.\n", __func__);
        }
        close(send_fd);

        if (producerStatus == 0 && consumerStatus == 0)
        {
            printf("&&&& EGLStream_CUDA_CrossGPU PASSED\n");
        }
        else
        {
            printf("&&&& EGLStream_CUDA_CrossGPU FAILED\n");
        }
    }
    else // Producer
    {
        test_cuda_producer_s cudaProducer;
        memset(&cudaProducer, 0, sizeof(test_cuda_producer_s));
        cudaProducer.profileAPI = profileAPIs;
        int producerStatus = 0;

        setenv("CUDA_EGL_PRODUCER_RETURN_WAIT_TIMEOUT", "1600", 0);

        int connect_fd = -1;
        // Hook up Ctrl-C handler
        signal(SIGINT, sig_handler);

        curesult = cudaDeviceCreateProducer(&cudaProducer);
        if (curesult != CUDA_SUCCESS) {
            producerStatus = -1;
            goto DoneProd;
        }

        args.charCnt = WIDTH * HEIGHT * 4;
        cuCtxPushCurrent(cudaProducer.context);
        curesult = cudaProducerInit(&cudaProducer, &args);
        if (curesult != CUDA_SUCCESS) {
            printf("Cuda Producer: Init failed, status: %d\n", curesult);
            producerStatus = -1;
            goto DoneProd;
        }

        //Create connection to Consumer
        connect_fd = UnixSocketCreate(SOCK_PATH);
        if (-1 == connect_fd) {
            printf("%s: Cuda Producer could not create socket: %s.\n",__func__, SOCK_PATH );
            producerStatus = -1;
            goto DoneProd;
        }

        // Get the file descriptor of the stream from the consumer process
        // and re-create the EGL stream from it
        fileDescriptor = EGLStreamReceivefd(connect_fd);
        if (-1 == fileDescriptor) {
            printf("%s: Cuda Producer could not receive EGL file descriptor \n",__func__);
            producerStatus = -1;
            goto DoneProd;
        }

        if (!eglSetupExtensions(isCrossDevice)) {
            printf("SetupExtentions failed \n");
            curesult = CUDA_ERROR_UNKNOWN;
            producerStatus = -1;
            goto DoneProd;
        }

        if (!EGLStreamInit(isCrossDevice, 0, fileDescriptor)) {
            printf("EGLStream Init failed.\n");
            producerStatus = -1;
            curesult = CUDA_ERROR_UNKNOWN;
            goto DoneProd;
        }

        cudaProducer.eglDisplay = g_producerEglDisplay;
        cudaProducer.eglStream  = g_producerEglStream;

        //wait for consumer to connect first
        int err = 0;
        int wait_loop = 0;
        EGLint streamState = 0;
        do {
            err = eglQueryStreamKHR(cudaProducer.eglDisplay, cudaProducer.eglStream,
                                    EGL_STREAM_STATE_KHR, &streamState);
            if ((0 != err)&&(EGL_STREAM_STATE_CONNECTING_KHR != streamState)) {
                sleep(1);
                wait_loop++;
            }
        } while ((wait_loop<10)&&(0 != err)&&(streamState != EGL_STREAM_STATE_CONNECTING_KHR));

        if ((0 == err)||(wait_loop >= 10)) {
            printf("%s: Cuda Producer eglQueryStreamKHR EGL_STREAM_STATE_KHR failed.\n",__func__);
            producerStatus = -1;
            goto DoneProd;
        }

        if (CUDA_SUCCESS != (curesult = cuEGLStreamProducerConnect(&(cudaProducer.cudaConn), cudaProducer.eglStream, WIDTH, HEIGHT))) {
            printf("Connect CUDA producer FAILED with error %d\n", curesult);
            producerStatus = -1;
            goto DoneProd;
        }

        printf("main - Cuda Producer and Consumer Initialized.\n");

        cudaProducerPrepareFrame(&cudaEgl1, cudaProducer.cudaPtr, args.charCnt);
        cudaProducerPrepareFrame(&cudaEgl2, cudaProducer.cudaPtr1, args.charCnt);

        j=0;
        for (j = 0;  j < NUMTRIALS; j++)
        {
            curesult = cudaProducerPresentFrame(&cudaProducer, cudaEgl1, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Producer Test failed for frame = %d with cuda error:%d\n", j+1, curesult);
                producerStatus = -1;
                goto DoneProd;
            }

            curesult = cudaProducerPresentFrame(&cudaProducer, cudaEgl2, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Producer Test failed for frame = %d with cuda error:%d\n", j+1, curesult);
                producerStatus = -1;
                goto DoneProd;
            }


            curesult = cudaProducerReturnFrame(&cudaProducer, cudaEgl1, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Producer Test failed for frame = %d with cuda error:%d\n", j+1, curesult);
                producerStatus = -1;
                goto DoneProd;
            }

            curesult = cudaProducerReturnFrame(&cudaProducer, cudaEgl2, j);
            if (curesult != CUDA_SUCCESS) {
                printf("Cuda Producer Test failed for frame = %d with cuda error:%d\n", j+1, curesult);
                producerStatus = -1;
                goto DoneProd;
            }
        }

        cuCtxSynchronize();
        err =  cudaGetValueMismatch();
        if (err != cudaSuccess) {
            printf("Prod: App failed with value mismatch\n");
            curesult = CUDA_ERROR_UNKNOWN;
            producerStatus = -1;
            goto DoneProd;
        }

        printf("Tear Down Start.....\n");
        if (!eglQueryStreamKHR(
                    cudaProducer.eglDisplay,
                    cudaProducer.eglStream,
                    EGL_STREAM_STATE_KHR,
                    &streamState)) {
            printf("Main, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
            curesult = CUDA_ERROR_UNKNOWN;
            producerStatus = -1;
            goto DoneProd;
        }

        if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
            if (CUDA_SUCCESS != (curesult = cudaProducerDeinit(&cudaProducer))) {
                printf("Producer Disconnect FAILED with %d\n", curesult);
                producerStatus = -1;
                goto DoneProd;
            }
        }

        unsetenv("CUDA_EGL_PRODUCER_RETURN_WAIT_TIMEOUT");

DoneProd:
        EGLStreamFini();
        if (-1 == send(connect_fd, (void *)&producerStatus, sizeof(int), 0)) {
            printf("%s: Cuda Producer could not send status to consumer.\n", __func__);
        }
        close(connect_fd);
    }

    return 0;
}
