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

#include <curand.h>
#include <string>

// RNGs
class RNG
{
    public:
        enum RngType {Pseudo, Quasi, ScrambledQuasi};
        RNG(unsigned long prngSeed, unsigned int qrngDimensions, unsigned int nSamples);
        virtual ~RNG();

        float getNextU01(void);
        void getInfoString(std::string &msg);
        void selectRng(RngType type);
        void resetSeed(void);
        void resetDimensions(void);
        void incrementDimensions(void);

    private:
        // Generators
        curandGenerator_t *m_pCurrent;
        curandGenerator_t m_prng;
        curandGenerator_t m_qrng;
        curandGenerator_t m_sqrng;

        // Parameters
        unsigned long m_prngSeed;
        unsigned int  m_qrngDimensions;

        // Batches
        const unsigned int m_nSamplesBatchTarget;
        unsigned int       m_nSamplesBatchActual;
        unsigned int       m_nSamplesRemaining;
        void generateBatch(void);

        // Helpers
        void updateDimensions(void);
        void setBatchSize(void);

        // Buffers
        float *m_h_samples;
        float *m_d_samples;

        static const unsigned int s_maxQrngDimensions;
};
