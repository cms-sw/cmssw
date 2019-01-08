#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<random>
#include<limits>

#include <cuda/api_wrappers.h>

template<typename T, int NBINS, int S, int DELTA>
__global__
void mykernel(T const * __restrict__  v, uint32_t N) {

  assert(v);
  assert(N==12000);

  if (threadIdx.x==0) printf("start kernel for %d data\n",N);

  using Hist = HistoContainer<T,NBINS,12000,S,uint16_t>;

  __shared__ Hist  hist;
  __shared__ typename Hist::Counter ws[32];

  for (auto j=threadIdx.x; j<Hist::totbins(); j+=blockDim.x) { hist.off[j]=0;}
  __syncthreads();

  for (auto j=threadIdx.x; j<N; j+=blockDim.x) hist.count(v[j]);
  __syncthreads();

  assert(0==hist.size());
  __syncthreads();

  
  hist.finalize(ws);
  __syncthreads();

  assert(N==hist.size());
  for (auto j=threadIdx.x; j<Hist::nbins(); j+=blockDim.x) assert(hist.off[j]<=hist.off[j+1]);
  __syncthreads();

  if (threadIdx.x<32) ws[threadIdx.x]=0;  // used by prefix scan...
  __syncthreads();

  for (auto j=threadIdx.x; j<N; j+=blockDim.x) hist.fill(v[j],j);
  __syncthreads();
  assert(0==hist.off[0]);
  assert(N==hist.size());


   for (auto j=threadIdx.x; j<hist.size()-1; j+=blockDim.x) {
     auto p = hist.begin()+j;
     assert((*p)<N);
     auto k1 = Hist::bin(v[*p]);
     auto k2 = Hist::bin(v[*(p+1)]);
     assert(k2>=k1);
   }

   for (auto i=threadIdx.x; i<hist.size(); i+=blockDim.x) {
      auto p = hist.begin()+i;
      auto j = *p;
      auto b0 = Hist::bin(v[j]);
      int tot=0;
      auto ftest =  [&](int k) {
         assert(k>=0 && k<N);
         ++tot;
      };
      forEachInWindow(hist,v[j],v[j],ftest);
      int rtot = hist.size(b0);
      assert(tot==rtot);
      tot=0;
      auto vm = int(v[j])-DELTA;
      auto vp = int(v[j])+DELTA;
      constexpr int vmax = NBINS!=128 ? NBINS*2-1 : std::numeric_limits<T>::max();
      vm = std::max(vm, 0);
      vm = std::min(vm,vmax);
      vp = std::min(vp,vmax);
      vp = std::max(vp, 0);
      assert(vp>=vm);
      forEachInWindow(hist, vm,vp, ftest);
      int bp = Hist::bin(vp);
      int bm = Hist::bin(vm);
      rtot = hist.end(bp)-hist.begin(bm);
      assert(tot==rtot);
   }


}

template<typename T, int NBINS=128, int S=8*sizeof(T), int DELTA=1000>
void go() {

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get();


  std::mt19937 eng;

  int rmin=std::numeric_limits<T>::min();
  int rmax=std::numeric_limits<T>::max();
  if (NBINS!=128) {
    rmin=0;
    rmax=NBINS*2-1;
  }



  std::uniform_int_distribution<T> rgen(rmin,rmax);

  
  constexpr int N=12000;
  T v[N];

  auto v_d = cuda::memory::device::make_unique<T[]>(current_device, N);
  assert(v_d.get());

  using Hist = HistoContainer<T,NBINS,N,S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' ' << (rmax-rmin)/Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' <<  int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;  

  for (int it=0; it<5; ++it) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);
    if (it==2) for (long long j = N/2; j < N/2+N/4; j++) v[j]=4;

    assert(v_d.get());
    assert(v);
    cuda::memory::copy(v_d.get(), v, N*sizeof(T));
    assert(v_d.get());
    cuda::launch(mykernel<T,NBINS,S,DELTA>,{1,256},v_d.get(),N);
  }

}

int main() {
  go<int16_t>();
  go<uint8_t,128,8,4>();
  go<uint16_t,313/2,9,4>();


  return 0;
}
