#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<random>
#include<limits>

template<typename T>
void go() {
  std::mt19937 eng;
  std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(),std::numeric_limits<T>::max());

  
  constexpr int N=12000;
  T v[N];

  using Hist = HistoContainer<T,7,8>;
  std::cout << "HistoContainer " << Hist::nbins() << ' ' << Hist::binSize() << std::endl;
  
  Hist h;
  for (int it=0; it<5; ++it) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);
    h.zero();
    for (long long j = 0; j < N; j++) h.fill(v[j],j);
    
    std::cout << "nspills " << h.nspills << std::endl;    

    auto verify = [&](uint32_t i, uint32_t k, uint32_t t1, uint32_t t2) {
      assert(t1<N); assert(t2<N);
      if ( T(v[t1]-v[t2])<=0) std::cout << "for " << i <<':'<< v[k] <<" failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    for (uint32_t i=0; i<Hist::nbins(); ++i) {
      if (0==h.n[i]) continue;
      auto k= *h.begin(i);
      assert(k<N);
      auto kl = h.bin(v[k]-T(1000));
      auto kh =	h.bin(v[k]+T(1000));
      assert(kl!=i);  assert(kh!=i);
      // std::cout << kl << ' ' << kh << std::endl;
      for (auto j=h.begin(kl); j<h.end(kl); ++j) verify(i,k,k,(*j));
      for (auto	j=h.begin(kh); j<h.end(kh); ++j) verify(i,k,(*j),k);
    }
  }

  for (long long j = 0; j < N; j++) {
    auto b0 = h.bin(v[j]);
    auto stot = h.endSpill()-h.beginSpill();
    int w=0;
    int tot=0;
    auto ftest = [&](int k) {
       assert(k>=0 && k<N);
       tot++;
    };
    forEachInBins(h,v[j],w,ftest);
    int rtot = h.end(b0)-h.begin(b0) + stot;
    assert(tot==rtot);
    w=1; tot=0;
    forEachInBins(h,v[j],w,ftest);
    int bp = b0+1;
    int bm = b0-1;
    if (bp<int(h.nbins())) rtot += h.end(bp)-h.begin(bp);
    if (bm>=0) rtot += h.end(bm)-h.begin(bm);
    assert(tot==rtot);
    w=2; tot=0;
    forEachInBins(h,v[j],w,ftest);
    bp++;
    bm--;
    if (bp<int(h.nbins())) rtot += h.end(bp)-h.begin(bp);
    if (bm>=0) rtot += h.end(bm)-h.begin(bm);
    assert(tot==rtot);
  }

}

int main() {
  go<int16_t>();

  return 0;
}
