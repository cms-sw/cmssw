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

#ifndef CUDASHAREDMEM_H
#define CUDASHAREDMEM_H

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          extern __shared__  T sdata[];
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//
// With this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          SharedMemory<T> sdata;
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//****************************************************************************

// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by making it abstract (i.e. with pure virtual methods).
template <typename T>
struct SharedMemory
{
    // Ensure that we won't compile any un-specialized types
    virtual __device__ T &operator*() = 0;
    virtual __device__ T &operator[](int i) = 0;
};

#define BUILD_SHAREDMEMORY_TYPE(t, n) \
    template <> \
    struct SharedMemory<t> \
    { \
        __device__ t &operator*() { extern __shared__ t n[]; return *n; } \
        __device__ t &operator[](int i) { extern __shared__ t n[]; return n[i]; } \
    }

BUILD_SHAREDMEMORY_TYPE(int,            s_int);
BUILD_SHAREDMEMORY_TYPE(unsigned int,   s_uint);
BUILD_SHAREDMEMORY_TYPE(char,           s_char);
BUILD_SHAREDMEMORY_TYPE(unsigned char,  s_uchar);
BUILD_SHAREDMEMORY_TYPE(short,          s_short);
BUILD_SHAREDMEMORY_TYPE(unsigned short, s_ushort);
BUILD_SHAREDMEMORY_TYPE(long,           s_long);
BUILD_SHAREDMEMORY_TYPE(unsigned long,  s_ulong);
BUILD_SHAREDMEMORY_TYPE(bool,           s_bool);
BUILD_SHAREDMEMORY_TYPE(float,          s_float);
BUILD_SHAREDMEMORY_TYPE(double,         s_double);

#endif
