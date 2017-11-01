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

#include "multithreading.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

//Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

//Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    WaitForMultipleObjects(num, threads, true, INFINITE);

    for (int i = 0; i < num; i++)
    {
        CloseHandle(threads[i]);
    }
}

//Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount)
{
    CUTBarrier barrier;

    InitializeCriticalSection(&barrier.criticalSection);
    barrier.barrierEvent = CreateEvent(NULL, TRUE, FALSE, TEXT("BarrierEvent"));
    barrier.count = 0;
    barrier.releaseCount = releaseCount;

    return barrier;
}

//Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier *barrier)
{
    int myBarrierCount;
    EnterCriticalSection(&barrier->criticalSection);
    myBarrierCount = ++barrier->count;
    LeaveCriticalSection(&barrier->criticalSection);

    if (myBarrierCount >= barrier->releaseCount)
    {
        SetEvent(barrier->barrierEvent);
    }
}

//Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier)
{
    WaitForSingleObject(barrier->barrierEvent, INFINITE);
}

//Destroy barrier
void cutDestroyBarrier(CUTBarrier *barrier)
{

}


#else
//Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    pthread_join(thread, NULL);
}

//Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    for (int i = 0; i < num; i++)
    {
        cutEndThread(threads[i]);
    }
}

//Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount)
{
    CUTBarrier barrier;

    barrier.count = 0;
    barrier.releaseCount = releaseCount;

    pthread_mutex_init(&barrier.mutex, 0);
    pthread_cond_init(&barrier.conditionVariable,0);


    return barrier;
}

//Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier *barrier)
{
    int myBarrierCount;
    pthread_mutex_lock(&barrier->mutex);
    myBarrierCount = ++barrier->count;
    pthread_mutex_unlock(&barrier->mutex);

    if (myBarrierCount >=barrier->releaseCount)
    {
        pthread_cond_signal(&barrier->conditionVariable);
    }
}

//Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier)
{
    pthread_mutex_lock(&barrier->mutex);

    while (barrier->count < barrier->releaseCount)
    {
        pthread_cond_wait(&barrier->conditionVariable, &barrier->mutex);
    }

    pthread_mutex_unlock(&barrier->mutex);
}

//Destroy barrier
void cutDestroyBarrier(CUTBarrier *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->conditionVariable);
}

#endif
