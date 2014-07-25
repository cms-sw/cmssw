#ifndef H5584ef3f37df484dc481a9489d08bca6
#define H5584ef3f37df484dc481a9489d08bca6
#include <atomic>
#include <algorithm>

namespace edm {
// keeps the running average of the last N entries
// thread safe, fast: does not garantee precise update in case of collision
  class RunningAverage {
  public:
    static constexpr int N = 16;  // better be a power of 2
    explicit RunningAverage(unsigned int k=4) : m_mean(N*k), curr(0) {
      for (auto & i : buffer) i=k; 
    }

    int mean() const { return m_mean/N;}

    int upper() const { auto lm = mean(); return lm+=(std::abs(buffer[0]-lm)+std::abs(buffer[8]-lm)); }  // about 2 sigma

   void update(unsigned int q) {
      int e=curr;
      while(!curr.compare_exchange_weak(e,e+1)); 
      int k = (N-1)&e;
      int old = buffer[k];
      if (!buffer[k].compare_exchange_strong(old,q)) return;
      m_mean+= (q-old); 
    }


private:
    std::atomic<int> buffer[N];
    std::atomic<int> m_mean;
    std::atomic<int> curr;

  };
}

#endif


