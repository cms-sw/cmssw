#include "HeterogeneousCore/CUDAUtilities/interface/AtomicPairCounter.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

__global__
void update(AtomicPairCounter * dc,  uint32_t * ind, uint32_t * cont,  uint32_t n) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;

  auto m = i%11;
  m = m%6 +1;  // max 6, no 0
  auto c = dc->add(m);
  assert(c.m<n);
  ind[c.m] = c.n;
  for(int j=c.n; j<c.n+m; ++j) cont[j]=i; 

};

__global__
void finalize(AtomicPairCounter const * dc,  uint32_t * ind, uint32_t * cont,  uint32_t n) {
  assert(dc->get().m==n);
  ind[n]= dc->get().n;
}

__global__
void verify(AtomicPairCounter const * dc, uint32_t const * ind, uint32_t const * cont,  uint32_t n) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
  assert(0==ind[0]);
  assert(dc->get().m==n);
  assert(ind[n] == dc->get().n);
  auto ib = ind[i];
  auto ie = ind[i+1];
  auto k = cont[ib++];
  assert(k<n);
  for (;ib<ie; ++ib) assert(cont[ib]==k);
}

#include<iostream>
int main() {

    AtomicPairCounter * dc_d;
    cudaMalloc(&dc_d, sizeof(AtomicPairCounter));
    cudaMemset(dc_d, 0, sizeof(AtomicPairCounter));

    std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

    constexpr uint32_t N=20000;
    constexpr uint32_t M=N*6;
    uint32_t *n_d, *m_d;
    cudaMalloc(&n_d, N*sizeof(int));
    // cudaMemset(n_d, 0, N*sizeof(int));
    cudaMalloc(&m_d, M*sizeof(int));


    update<<<2000, 512 >>>(dc_d,n_d,m_d,10000);
    finalize<<<1,1 >>>(dc_d,n_d,m_d,10000);
    verify<<<2000, 512 >>>(dc_d,n_d,m_d,10000);

    AtomicPairCounter dc;
    cudaMemcpy(&dc, dc_d, sizeof(AtomicPairCounter), cudaMemcpyDeviceToHost);

    std::cout << dc.get().n << ' ' << dc.get().m << std::endl;

    return 0;
}
