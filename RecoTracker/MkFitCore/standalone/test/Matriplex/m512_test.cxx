#include "immintrin.h"

#include <cstdio>

const int NN = 64;

#define LD(a, i)      _mm512_load_ps(&a[i*16])
#define ADD(a, b)     _mm512_add_ps(a, b) 
#define MUL(a, b)     _mm512_mul_ps(a, b)
#define FMA(a, b, v)  _mm512_fmadd_ps(a, b, v)
#define ST(a, i, r)   _mm512_store_ps(&a[i*16], r)

// Can even be global!
__m512 all_ones = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

int main()
{
  float *p = (float*) std::aligned_alloc(64, NN*sizeof(float));
  float *q = (float*) std::aligned_alloc(64, NN*sizeof(float));

  for (int i = 0; i < NN; ++i)
  {
    p[i] = i;
  }

  __m512 a = LD(p, 0);
  __m512 b = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };//LD(p, 1);

  b = all_ones;

  __m512 c = ADD(a, b);

  ST(q, 0, c);

  for (int i = 0; i < 16; ++i)
  {
    printf("%2d %4.0f %4.0f %4.0f\n", i, p[i], p[i+16], q[i]);
  }

  std::free(p);
  std::free(q);

  return 0;
}
