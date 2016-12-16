#ifndef RecoParticleFlow_Benchmark_BenchmarkTree_h
#define RecoParticleFlow_Benchmark_BenchmarkTree_h


#include <TTree.h>

class BenchmarkTreeEntry {

 public:
  BenchmarkTreeEntry() : 
    deltaEt(999), 
    deltaEta(-9),
    eta(-10),
    et(-1)
    {}

  BenchmarkTreeEntry& operator=(const BenchmarkTreeEntry& other) {
    deltaEt= other.deltaEt;
    deltaEta= other.deltaEta;
    eta= other.eta;
    et= other.et;

    return *this;
  }
  
  float deltaEt;
  float deltaEta;
  float eta;
  float et;
};


class BenchmarkTree : public TTree {
  
 public:
  BenchmarkTree( const char* name,
			const char* title);
  using TTree::Fill;
  void Fill( const BenchmarkTreeEntry& entry );
  
 private:
  BenchmarkTreeEntry* entry_;
};

#endif
