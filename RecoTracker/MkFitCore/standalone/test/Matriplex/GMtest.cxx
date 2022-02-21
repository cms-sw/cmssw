#include "Math/SMatrix.h"

#include "MatriplexSym.h"

#include <random>

/*
# Generate .ah files (make sure DIM, DOM and pattern match):
  ./GMtest.pl
# Compile:
  icc -std=gnu++11 -openmp -mavx -O3 -I.. -I../.. GMtest.cxx -o GMtest
*/

typedef long long long64;

const int N   = 16;

const int DIM =  3;
const int DOM =  6;

#ifdef MPLEX_INTRINSICS
# if defined(__AVX512F__)
#   warning "MPLEX_INTRINSICS CMP_EPS = 2e-7 --> 3e-7"
const float CMP_EPS = 3e-7;
# elif defined(__AVX__)
#   warning "MPLEX_INTRINSICS CMP_EPS = 2e-7 --> 5e-7"
const float CMP_EPS = 5e-7;
# else
#   warning "MPLEX_INTRINSICS CMP_EPS = 2e-7"
const float CMP_EPS = 2e-7;
# endif
#else
# if defined(__AVX512F__)
#   warning "NO MPLEX_INTRINSICS CMP_EPS = 4e-7"
const float CMP_EPS = 4e-7;
# else
#   warning "NO MPLEX_INTRINSICS CMP_EPS = 4e-7 --> 5e-7"
const float CMP_EPS = 5e-7;
# endif
#endif

typedef ROOT::Math::SMatrix<float, DIM, DOM>                                     SMatX;
typedef ROOT::Math::SMatrix<float, DOM, DIM>                                     SMatXT;
typedef ROOT::Math::SMatrix<float, DIM, DIM, ROOT::Math::MatRepSym<float, DIM> > SMatS;

typedef Matriplex::Matriplex   <float, DIM, DOM, N>   MPlexX;
typedef Matriplex::Matriplex   <float, DOM, DIM, N>   MPlexXT;
typedef Matriplex::MatriplexSym<float, DIM,      N>   MPlexS;

void Multify(const MPlexS& A, const MPlexX& B, MPlexX& C)
{
   // C = A * B

   typedef float T;

   const T *a = A.fArray; __assume_aligned(a, 64);
   const T *b = B.fArray; __assume_aligned(b, 64);
         T *c = C.fArray; __assume_aligned(c, 64);

#include "multify.ah"
}

void MultifyTranspose(const MPlexS& A, const MPlexX& B, MPlexXT& C)
{
   // C = BT * A;

   typedef float T;

   const T *a = A.fArray; __assume_aligned(a, 64);
   const T *b = B.fArray; __assume_aligned(b, 64);
         T *c = C.fArray; __assume_aligned(c, 64);

#include "multify-transpose.ah"
}

int main()
{
  SMatS   a[N];
  SMatX   b[N],  c[N];
  SMatXT  bt[N], ct[N];

  MPlexS  A;
  MPlexX  B, C;
  MPlexXT CT;

  std::default_random_engine      gen(0xbeef0133);
  std::normal_distribution<float> dis(1.0, 0.05);

  long64 count = 1;

init:

  for (int m = 0; m < N; ++m)
  {
    for (int i = 0; i < 3; ++i)
    {
      for (int j = i; j < 6; ++j)
      {
        if (j < DIM)  a[m](i,j) = dis(gen);

        b[m](i,j) = dis(gen);
      }
    }

    // Enforce pattern from GMtest.pl
    a[m](1, 1) = 1;
    b[m](0, 4) = 0;
    b[m](1, 1) = 1;
    b[m](1, 3) = 1;
    b[m](1, 4) = 0;
    b[m](2, 4) = 0;

    A.CopyIn(m, a[m].Array());
    B.CopyIn(m, b[m].Array());

    c[m]  = a[m] * b[m];

    bt[m] = ROOT::Math::Transpose(b[m]);
    ct[m] = bt[m] * a[m];
  }

  Multify(A, B, C);
  MultifyTranspose(A, B, CT);

  for (int m = 0; m < N; ++m)
  {
    bool dump = false;

    for (int j = 0; j < DIM; ++j)
    {
      for (int k = 0; k < DOM; ++k)
      {
        // There are occasional diffs up to 4.768372e-07 on host, very very
        // rarely on MIC. Apparently this is a rounding difference between AVX
        // and normal maths. On MIC it might be usage of FMA?
        // The above was for 3x3.
        // For 6x6 practically all elements differ by 4.768372e-07, some
        // by 9.536743e-07.
        if (std::abs(c[m](j,k) - C.At(m, j, k)) > CMP_EPS)
        {
          dump = true;
          printf("MULTIFY   M=%d  %d,%d d=%e (count = %lld)\n", m, j, k, c[m](j,k) - C.At(m, j, k), count);
        }
      }
    }

    if (dump && false)
    {
      printf("\n");
      for (int i = 0; i < DIM; ++i)
      {
        for (int j = 0; j < DOM; ++j)
          printf("%8f ", c[m](i,j));
        printf("\n");
      }
      printf("\n");

      for (int i = 0; i < DIM; ++i)
      {
        for (int j = 0; j < DOM; ++j)
          printf("%8f ", C.At(m, i, j));
        printf("\n");
      }
      printf("\n");
    }
    if (dump)
    {
      printf("\n");
    }
  }

  // Shameless cut-n-paste of above dump for transpose check with minor changes.
  // Should make a function, I know ... but ... no time to lose.

  for (int m = 0; m < N; ++m)
  {
    bool dump = false;

    for (int j = 0; j < DOM; ++j)
    {
      for (int k = 0; k < DIM; ++k)
      {
        // There are occasional diffs up to 4.768372e-07 on host, very very
        // rarely on MIC. Apparently this is a rounding difference between AVX
        // and normal maths. On MIC it might be usage of FMA?
        // The above was for 3x3.
        // For 6x6 practically all elements differ by 4.768372e-07, some
        // by 9.536743e-07.
        if (std::abs(ct[m](j,k) - CT.At(m, j, k)) > CMP_EPS)
        {
          dump = true;
          printf("TRANSPOSE M=%d  %d,%d d=%e (count = %lld)\n", m, j, k, ct[m](j,k) - CT.At(m, j, k), count);
        }
      }
    }

    if (dump && false)
    {
      printf("\n");
      for (int i = 0; i < DOM; ++i)
      {
        for (int j = 0; j < DIM; ++j)
          printf("%8f ", ct[m](i,j));
        printf("\n");
      }
      printf("\n");

      for (int i = 0; i < DIM; ++i)
      {
        for (int j = 0; j < DOM; ++j)
          printf("%8f ", CT.At(m, i, j));
        printf("\n");
      }
      printf("\n");
    }
    if (dump)
    {
      printf("\n");
    }
  }


  ++count;
  goto init;

  return 0;
}
