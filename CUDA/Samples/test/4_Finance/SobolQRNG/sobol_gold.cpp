/*
* Portions Copyright (c) 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
* Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
* Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
*
* Sobol Quasi-random Number Generator example
*
* Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
* http://people.maths.ox.ac.uk/~gilesm/
*
* and C code developed by Stephen Joe, University of Waikato, New Zealand
* and Frances Kuo, University of New South Wales, Australia
* http://web.maths.unsw.edu.au/~fkuo/sobol/
*
* For theoretical background see:
*
* P. Bratley and B.L. Fox.
* Implementing Sobol's quasirandom sequence generator
* http://portal.acm.org/citation.cfm?id=42288
* ACM Trans. on Math. Software, 14(1):88-100, 1988
*
* S. Joe and F. Kuo.
* Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
* http://portal.acm.org/citation.cfm?id=641879
* ACM Trans. on Math. Software, 29(1):49-57, 2003
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "sobol.h"
#include "sobol_gold.h"
#include "sobol_primitives.h"

#define k_2powneg32 2.3283064E-10F

// Windows does not provide ffs (find first set) so here is a
// fairly simple implementation.
// WIN32 is defined on 32 and 64 bit Windows
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
int ffs(const unsigned int &i)
{
    unsigned int v = i;
    unsigned int count;

    if (! v)
    {
        count = 0;
    }
    else
    {
        count = 2;

        if ((v & 0xffff) == 0)
        {
            v >>= 16;
            count += 16;
        }

        if ((v & 0xff) == 0)
        {
            v >>= 8;
            count += 8;
        }

        if ((v & 0xf) == 0)
        {
            v >>= 4;
            count += 4;
        }

        if ((v & 0x3) == 0)
        {
            v >>= 2;
            count += 2;
        }

        count -= v & 0x1;
    }

    return count;
}
#endif

// Create the direction numbers, based on the primitive polynomials.
void initSobolDirectionVectors(int n_dimensions, unsigned int *directions)
{
    unsigned int *v = directions;

    for (int dim = 0 ; dim < n_dimensions ; dim++)
    {
        // First dimension is a special case
        if (dim == 0)
        {
            for (int i = 0 ; i < n_directions ; i++)
            {
                // All m's are 1
                v[i] = 1 << (31 - i);
            }
        }
        else
        {
            int d = sobol_primitives[dim].degree;

            // The first direction numbers (up to the degree of the polynomial)
            // are simply v[i] = m[i] / 2^i (stored in Q0.32 format)
            for (int i = 0 ; i < d ; i++)
            {
                v[i] = sobol_primitives[dim].m[i] << (31 - i);
            }

            // The remaining direction numbers are computed as described in
            // the Bratley and Fox paper.
            // v[i] = a[1]v[i-1] ^ a[2]v[i-2] ^ ... ^ a[v-1]v[i-d+1] ^ v[i-d] ^ v[i-d]/2^d
            for (int i = d ; i < n_directions ; i++)
            {
                // First do the v[i-d] ^ v[i-d]/2^d part
                v[i] = v[i - d] ^ (v[i - d] >> d);

                // Now do the a[1]v[i-1] ^ a[2]v[i-2] ^ ... part
                // Note that the coefficients a[] are zero or one and for compactness in
                // the input tables they are stored as bits of a single integer. To extract
                // the relevant bit we use right shift and mask with 1.
                // For example, for a 10 degree polynomial there are ten useful bits in a,
                // so to get a[2] we need to right shift 7 times (to get the 8th bit into
                // the LSB) and then mask with 1.
                for (int j = 1 ; j < d ; j++)
                {
                    v[i] ^= (((sobol_primitives[dim].a >> (d - 1 - j)) & 1) * v[i - j]);
                }
            }
        }

        v += n_directions;
    }
}

// Reference model for generating Sobol numbers on the host
void sobolCPU(int n_vectors, int n_dimensions, unsigned int *directions, float *output)
{
    unsigned int *v = directions;

    for (int d = 0 ; d < n_dimensions ; d++)
    {
        unsigned int X = 0;
        // x[0] is zero (in all dimensions)
        output[n_vectors * d] = 0.0;

        for (int i = 1 ; i < n_vectors ; i++)
        {
            // x[i] = x[i-1] ^ v[c]
            //  where c is the index of the rightmost zero bit in i
            //  minus 1 (since C arrays count from zero)
            // In the Bratley and Fox paper this is equation (**)
            X ^= v[ffs(~(i - 1)) - 1];
            output[i + n_vectors * d] = (float)X * k_2powneg32;
        }

        v += n_directions;
    }
}
