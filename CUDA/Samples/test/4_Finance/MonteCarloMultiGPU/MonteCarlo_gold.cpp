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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <curand.h>

//#include "curand_kernel.h"
#include "helper_cuda.h"



////////////////////////////////////////////////////////////////////////////////
// Common types
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_common.h"


////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for Monte Carlo results validation
////////////////////////////////////////////////////////////////////////////////
#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.39894228040143267793994605993438

//Polynomial approximation of
//cumulative normal distribution function
double CND(double d)
{
    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

//Black-Scholes formula for call value
extern "C" void BlackScholesCall(
    float &callValue,
    TOptionData optionData
)
{
    double     S = optionData.S;
    double     X = optionData.X;
    double     T = optionData.T;
    double     R = optionData.R;
    double     V = optionData.V;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);
    double expRT = exp(- R * T);

    callValue = (float)(S * CNDD1 - X * expRT * CNDD2);
}


////////////////////////////////////////////////////////////////////////////////
// CPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////
static double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

extern "C" void MonteCarloCPU(
    TOptionValue    &callValue,
    TOptionData optionData,
    float *h_Samples,
    int pathN
)
{
    const double        S = optionData.S;
    const double        X = optionData.X;
    const double        T = optionData.T;
    const double        R = optionData.R;
    const double        V = optionData.V;
    const double    MuByT = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);

    float *samples;
    curandGenerator_t gen;

    checkCudaErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    unsigned long long seed = 1234ULL;
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,  seed));

    if (h_Samples != NULL)
    {
        samples = h_Samples;
    }
    else
    {
        samples = (float *) malloc(pathN * sizeof(float));
        checkCudaErrors(curandGenerateNormal(gen, samples, pathN, 0.0, 1.0));
    }

    // for(int i=0; i<10; i++) printf("CPU sample = %f\n", samples[i]);

    double sum = 0, sum2 = 0;

    for (int pos = 0; pos < pathN; pos++)
    {

        double    sample = samples[pos];
        double callValue = endCallValue(S, X, sample, MuByT, VBySqrtT);
        sum  += callValue;
        sum2 += callValue * callValue;
    }

    if (h_Samples == NULL) free(samples);

    checkCudaErrors(curandDestroyGenerator(gen));

    //Derive average from the total sum and discount by riskfree rate
    callValue.Expected = (float)(exp(-R * T) * sum / (double)pathN);
    //Standard deviation
    double stdDev = sqrt(((double)pathN * sum2 - sum * sum)/ ((double)pathN * (double)(pathN - 1)));
    //Confidence width; in 95% of all cases theoretical value lies within these borders
    callValue.Confidence = (float)(exp(-R * T) * 1.96 * stdDev / sqrt((double)pathN));
}
