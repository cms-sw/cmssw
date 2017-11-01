/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_STOPWATCH_H
#define NV_STOPWATCH_H

#include "StopWatchBase.h"

// include OS specific policy
#ifdef WIN32
#include "StopWatchWin.h"
namespace npp
{
    typedef StopWatchBase<StopWatchWin>  StopWatch;
} // npp namesapce
#else
#include "StopWatchLinux.h"
namespace npp
{
    typedef StopWatchBase<StopWatchLinux>  StopWatch;
} // npp namesapce
#endif


#endif // NV_STOPWATCH_H

