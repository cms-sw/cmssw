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


#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (4 * block_size) // Matrix A width
#define HA (6 * block_size) // Matrix A height
#define WB (4 * block_size) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

#endif // _MATRIXMUL_H_
