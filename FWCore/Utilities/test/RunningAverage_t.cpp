#define private public
#include "FWCore/Utilities/interface/RunningAverage.h"
#undef private

namespace {

  edm::RunningAverage localRA;

}

#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <random>
#include <algorithm>
#include <type_traits>

int main() {


  //tbb::task_scheduler_init init;  // Automatic number of threads
  tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());  // Explicit number of threads

  // std::random_device rd;
  std::mt19937 e2; // (rd());
  std::normal_distribution<> normal_dist(1000., 200.);



  thread_local std::vector<float> v;  

  

  int kk=0;


  int n=2000;

  int res[n];
  int qq[n]; for (auto & y:qq) y = std::max(0.,normal_dist(e2));

  auto theLoop = [&](int i) {
    kk++;
    v.reserve(res[i]=localRA.upper());
    v.resize(qq[i]);
    localRA.update(v.size());
    decltype(v) t;
    swap(v,t);
  };


  tbb::parallel_for(
		    tbb::blocked_range<size_t>(0,n),
		    [&](const tbb::blocked_range<size_t>& r) {
		      for (size_t i=r.begin();i<r.end();++i) theLoop(i);
    }
  );


  auto mm = std::max_element(res,res+n);
  std::cout << kk << ' ' << localRA.m_curr << ' ' << localRA.mean() << std::endl;
  for (auto & i : localRA.m_buffer) std::cout << i << ' ';
  std::cout << std::endl;
  std::cout << std::accumulate(res,res+n,0)/n 
	    << ' ' << *std::min_element(res+16,res+n) << ',' << *mm << std::endl;

  return 0;


}

