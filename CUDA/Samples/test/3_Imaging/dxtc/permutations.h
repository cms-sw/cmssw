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

#ifndef PERMUTATIONS_H
#define PERMUTATIONS_H

#include <helper_cuda.h> // assert

static void computePermutations(uint permutations[1024])
{
    int indices[16];
    int num = 0;

    // 3 element permutations:

    // first cluster [0,i) is at the start
    for (int m = 0; m < 16; ++m)
    {
        indices[m] = 0;
    }

    const int imax = 15;

    for (int i = imax; i >= 0; --i)
    {
        // second cluster [i,j) is half along
        for (int m = i; m < 16; ++m)
        {
            indices[m] = 2;
        }

        const int jmax = (i == 0) ? 15 : 16;

        for (int j = jmax; j >= i; --j)
        {
            // last cluster [j,k) is at the end
            if (j < 16)
            {
                indices[j] = 1;
            }

            uint permutation = 0;

            for (int p = 0; p < 16; p++)
            {
                permutation |= indices[p] << (p * 2);
                //permutation |= indices[15-p] << (p * 2);
            }

            permutations[num] = permutation;

            num++;
        }
    }

    assert(num == 151);

    for (int i = 0; i < 9; i++)
    {
        permutations[num] = 0x000AA555;
        num++;
    }

    assert(num == 160);

    // Append 4 element permutations:

    // first cluster [0,i) is at the start
    for (int m = 0; m < 16; ++m)
    {
        indices[m] = 0;
    }

    for (int i = imax; i >= 0; --i)
    {
        // second cluster [i,j) is one third along
        for (int m = i; m < 16; ++m)
        {
            indices[m] = 2;
        }

        const int jmax = (i == 0) ? 15 : 16;

        for (int j = jmax; j >= i; --j)
        {
            // third cluster [j,k) is two thirds along
            for (int m = j; m < 16; ++m)
            {
                indices[m] = 3;
            }

            int kmax = (j == 0) ? 15 : 16;

            for (int k = kmax; k >= j; --k)
            {
                // last cluster [k,n) is at the end
                if (k < 16)
                {
                    indices[k] = 1;
                }

                uint permutation = 0;

                bool hasThree = false;

                for (int p = 0; p < 16; p++)
                {
                    permutation |= indices[p] << (p * 2);
                    //permutation |= indices[15-p] << (p * 2);

                    if (indices[p] == 3) hasThree = true;
                }

                if (hasThree)
                {
                    permutations[num] = permutation;
                    num++;
                }
            }
        }
    }

    assert(num == 975);

    // 1024 - 969 - 7 = 48 extra elements

    // It would be nice to set these extra elements with better values...
    for (int i = 0; i < 49; i++)
    {
        permutations[num] = 0x00AAFF55;
        num++;
    }

    assert(num == 1024);
}


#endif // PERMUTATIONS_H
