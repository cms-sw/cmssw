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
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/

// Utilities and System includes


// Includes
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include "rng.h"

// Shared Library Test Functions
#include <helper_timer.h>
#include <helper_cuda.h>

const unsigned int RNG::s_maxQrngDimensions = 20000;

RNG::RNG(unsigned long prngSeed, unsigned int qrngDimensions, unsigned int nSamples)
    : m_prngSeed(prngSeed),
      m_qrngDimensions(qrngDimensions),
      m_nSamplesBatchTarget(nSamples),
      m_nSamplesRemaining(0)
{
    using std::string;
    using std::runtime_error;
    using std::invalid_argument;

    if (m_prngSeed == 0)
    {
        throw invalid_argument("PRNG seed must be non-zero");
    }

    if (m_qrngDimensions == 0)
    {
        throw invalid_argument("QRNG dimensions must be non-zero");
    }

    if (m_nSamplesBatchTarget == 0)
    {
        throw invalid_argument("RNG batch size must be non-zero");
    }

    if (m_nSamplesBatchTarget < s_maxQrngDimensions)
    {
        throw invalid_argument("RNG batch size must be greater than RNG::s_maxQrngDimensions");
    }

    curandStatus_t curandResult;
    cudaError_t    cudaResult;

    // Allocate sample array in host mem
    m_h_samples = (float *)malloc(m_nSamplesBatchTarget * sizeof(float));

    if (m_h_samples == NULL)
    {
        throw runtime_error("Could not allocate host memory for RNG::m_h_samples");
    }

    // Allocate sample array in device mem
    cudaResult = cudaMalloc((void **)&m_d_samples, m_nSamplesBatchTarget * sizeof(float));

    if (cudaResult != cudaSuccess)
    {
        string msg("Could not allocate device memory for RNG::m_d_samples: ");
        msg += cudaGetErrorString(cudaResult);
        throw runtime_error(msg);
    }

    // Create the Random Number Generators
    curandResult = curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_XORWOW);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not create pseudo-random number generator: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    curandResult = curandCreateGenerator(&m_qrng, CURAND_RNG_QUASI_SOBOL32);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not create quasi-random number generator: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    curandResult = curandCreateGenerator(&m_sqrng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not create scrambled quasi-random number generator: ");
        msg += curandResult;
        throw runtime_error(msg);
    }


    // Setup initial parameters
    resetSeed();
    updateDimensions();
    setBatchSize();

    // Set default RNG to be pseudo-random (XORWOW)
    m_pCurrent = &m_prng;
}

RNG::~RNG()
{
    curandDestroyGenerator(m_prng);
    curandDestroyGenerator(m_qrng);
    curandDestroyGenerator(m_sqrng);

    if (m_d_samples)
    {
        cudaFree(m_d_samples);
    }

    if (m_h_samples)
    {
        free(m_h_samples);
    }
}

void RNG::generateBatch(void)
{
    using std::string;
    using std::runtime_error;

    cudaError_t    cudaResult;
    curandStatus_t curandResult;

    // Generate random numbers
    curandResult = curandGenerateUniform(*m_pCurrent, m_d_samples, m_nSamplesBatchActual);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not generate random numbers: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    // Copy random numbers to host
    cudaResult = cudaMemcpy(m_h_samples, m_d_samples, m_nSamplesBatchActual * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaResult != cudaSuccess)
    {
        string msg("Could not copy random numbers to host: ");
        msg += cudaGetErrorString(cudaResult);
        throw runtime_error(msg);
    }
}

float RNG::getNextU01(void)
{
    if (m_nSamplesRemaining == 0)
    {
        generateBatch();
        m_nSamplesRemaining = m_nSamplesBatchActual;
    }

    if (m_pCurrent == &m_prng)
    {
        return m_h_samples[m_nSamplesBatchActual - m_nSamplesRemaining--];
    }
    else
    {
        unsigned int index         = m_nSamplesBatchActual - m_nSamplesRemaining--;
        unsigned int samplesPerDim = m_nSamplesBatchActual / m_qrngDimensions;
        unsigned int dimOffset     = (index % m_qrngDimensions) * samplesPerDim;
        unsigned int drawOffset    = index / m_qrngDimensions;
        return m_h_samples[dimOffset + drawOffset];
    }
}

void RNG::getInfoString(std::string &msg)
{
    using std::stringstream;

    stringstream ss;

    if (m_pCurrent == &m_prng)
    {
        ss << "XORWOW (seed=" << m_prngSeed << ")";
    }
    else if (m_pCurrent == &m_qrng)
    {
        ss << "Sobol (dimensions=" << m_qrngDimensions << ")";
    }
    else if (m_pCurrent == &m_sqrng)
    {
        ss << "Scrambled Sobol (dimensions=" << m_qrngDimensions << ")";
    }
    else
    {
        ss << "Invalid RNG";
    }

    msg.assign(ss.str());
}

void RNG::selectRng(RNG::RngType type)
{
    switch (type)
    {
        case Quasi:
            m_pCurrent = &m_qrng;
            break;

        case ScrambledQuasi:
            m_pCurrent = &m_sqrng;
            break;

        case Pseudo:
        default:
            m_pCurrent = &m_prng;
            break;
    }

    setBatchSize();
}

void RNG::resetSeed(void)
{
    using std::runtime_error;

    curandStatus_t curandResult;
    curandResult = curandSetPseudoRandomGeneratorSeed(m_prng, m_prngSeed);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        std::string msg("Could not set pseudo-random number generator seed: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    curandResult = curandSetGeneratorOffset(m_prng, 0);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        std::string msg("Could not set pseudo-random number generator offset: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    setBatchSize();
}

void RNG::resetDimensions(void)
{
    m_qrngDimensions = 1;
    updateDimensions();
    setBatchSize();
}

void RNG::incrementDimensions(void)
{
    if (++m_qrngDimensions > s_maxQrngDimensions)
    {
        m_qrngDimensions = 1;
    }

    updateDimensions();
    setBatchSize();
}

void RNG::updateDimensions(void)
{
    using std::runtime_error;

    curandStatus_t curandResult;
    curandResult = curandSetQuasiRandomGeneratorDimensions(m_qrng, m_qrngDimensions);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        std::string msg("Could not set quasi-random number generator dimensions: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    curandResult = curandSetGeneratorOffset(m_qrng, 0);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        std::string msg("Could not set quasi-random number generator offset: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    curandResult = curandSetQuasiRandomGeneratorDimensions(m_sqrng, m_qrngDimensions);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        std::string msg("Could not set scrambled quasi-random number generator dimensions: ");
        msg += curandResult;
        throw runtime_error(msg);
    }

    curandResult = curandSetGeneratorOffset(m_sqrng, 0);

    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        std::string msg("Could not set scrambled quasi-random number generator offset: ");
        msg += curandResult;
        throw runtime_error(msg);
    }
}

void RNG::setBatchSize(void)
{
    if (m_pCurrent == &m_prng)
    {
        m_nSamplesBatchActual = m_nSamplesBatchTarget;
    }
    else
    {
        m_nSamplesBatchActual = (m_nSamplesBatchTarget / m_qrngDimensions) * m_qrngDimensions;
    }

    m_nSamplesRemaining = 0;
}
