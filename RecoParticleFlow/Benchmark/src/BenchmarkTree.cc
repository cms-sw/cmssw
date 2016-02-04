#include "RecoParticleFlow/Benchmark/interface/BenchmarkTree.h"



BenchmarkTree::BenchmarkTree( const char* name,
			      const char* title) 
  : TTree( name, title ), 
    entry_( new BenchmarkTreeEntry ) {
 
  Branch( "benchmarkEntry","BenchmarkTreeEntry", &entry_ );
}

void BenchmarkTree::Fill( const BenchmarkTreeEntry& entry ) {
  *entry_ = entry;
  TTree::Fill();
}
