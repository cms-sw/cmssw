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
    for (long long j = 0; j < N; j++) h.fill(v,j);
    
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

}

int main() {
  go<int16_t>();

  return 0;
}
