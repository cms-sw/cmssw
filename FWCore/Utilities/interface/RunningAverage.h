#ifndef FWCore_Utilities_RunningAverage_H
#define FWCore_Utilities_RunningAverage_H
#include <atomic>
#include <algorithm>
#include <array>

namespace edm {
// keeps the running average of the last N entries
// thread safe, fast: does not garantee precise update in case of collision
  class RunningAverage {
  public:
    static constexpr int N = 16;  // better be a power of 2
    explicit RunningAverage(unsigned int k=4) : m_mean(N*k), m_curr(0) {
      for (auto & i : m_buffer) i=k; 
    }

    int mean() const { return m_mean/N;}

    int upper() const { auto lm = mean(); return lm+=(std::abs(m_buffer[0]-lm)+std::abs(m_buffer[N/2]-lm)); }  // about 2 sigma

   void update(unsigned int q) {
      int e=m_curr;
      while(!m_curr.compare_exchange_weak(e,e+1)); 
      int k = (N-1)&e;
      int old = m_buffer[k];
      if (!m_buffer[k].compare_exchange_strong(old,q)) return;
      m_mean+= (q-old); 
    }


private:
    std::array<std::atomic<int>,N> m_buffer;
    std::atomic<int> m_mean;
    std::atomic<int> m_curr;

  };
}

#endif


