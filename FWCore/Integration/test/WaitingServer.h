#ifndef FWCore_Integration_WaitingServer_h
#define FWCore_Integration_WaitingServer_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace edmtest {
  namespace test_acquire {

    struct StreamData {
      StreamData() : in_(nullptr), out_(nullptr) {}

      std::vector<int> const* in_;
      std::vector<int>* out_;
      edm::WaitingTaskWithArenaHolder holder_;
    };

    class WaitingServer {
    public:
      WaitingServer(unsigned int iNumberOfStreams,
                    unsigned int iMinNumberOfStreamsBeforeDoingWork,
                    unsigned int iSecondsToWait)
          : m_perStream(iNumberOfStreams),
            m_minNumStreamsBeforeDoingWork(iMinNumberOfStreamsBeforeDoingWork),
            m_secondsToWait(iSecondsToWait),
            m_shouldStop(false) {}

      void start();
      void stop();

      void requestValuesAsync(unsigned int dataID,
                              std::vector<int> const* iIn,
                              std::vector<int>* iOut,
                              edm::WaitingTaskWithArenaHolder holder);

    private:
      void serverDoWork();

      bool readyForWork() const;

      std::mutex m_mutex;  //needed by m_cond
      std::condition_variable m_cond;
      std::unique_ptr<std::thread> m_thread;
      std::vector<StreamData> m_perStream;
      std::vector<unsigned int> m_waitingStreams;
      const unsigned int m_minNumStreamsBeforeDoingWork;
      const unsigned int m_secondsToWait;
      std::atomic<bool> m_shouldStop;
    };

    class Cache {
    public:
      std::vector<int> const& retrieved() const { return m_retrieved; }
      std::vector<int>& retrieved() { return m_retrieved; }

      std::vector<int> const& processed() const { return m_processed; }
      std::vector<int>& processed() { return m_processed; }

    private:
      std::vector<int> m_retrieved;
      std::vector<int> m_processed;
    };
  }  // namespace test_acquire
}  // namespace edmtest
#endif
