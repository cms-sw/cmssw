
#include "WaitingServer.h"

#include <exception>

namespace edmtest {
  namespace test_acquire {

    void WaitingServer::requestValuesAsync(unsigned int dataID,
                                           std::vector<int> const* iIn,
                                           std::vector<int>* iOut,
                                           edm::WaitingTaskWithArenaHolder holder) {
      auto& streamData = m_perStream.at(dataID);

      streamData.in_ = iIn;
      streamData.out_ = iOut;
      streamData.holder_ = std::move(holder);
      {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_waitingStreams.push_back(dataID);
      }
      m_cond.notify_one();  //wakes up the server thread
    }

    void WaitingServer::start() {
      m_thread = std::make_unique<std::thread>([this]() { serverDoWork(); });
    }

    void WaitingServer::stop() {
      m_shouldStop = true;
      if (m_thread) {
        m_thread->join();
        m_thread.reset();
      }
    }

    bool WaitingServer::readyForWork() const { return m_waitingStreams.size() >= m_minNumStreamsBeforeDoingWork; }

    void WaitingServer::serverDoWork() {
      while (not m_shouldStop) {
        std::vector<unsigned int> streamsToUse;
        {
          std::unique_lock<std::mutex> lk(m_mutex);

          m_cond.wait_for(lk, std::chrono::seconds(m_secondsToWait), [this]() -> bool { return readyForWork(); });

          // Once we know which streams have given us data
          // we can release the lock and let other streams
          // set their data
          streamsToUse.swap(m_waitingStreams);
          lk.unlock();
        }

        // Here is the work that the server does for the modules
        // it will just add 1 to each value it has been given
        for (auto index : streamsToUse) {
          auto& streamData = m_perStream.at(index);

          std::exception_ptr exceptionPtr;
          try {
            for (auto v : *streamData.in_) {
              streamData.out_->push_back(v + 1);
            }
          } catch (...) {
            exceptionPtr = std::current_exception();
          }
          streamData.holder_.doneWaiting(exceptionPtr);
        }
      }
    }
  }  // namespace test_acquire
}  // namespace edmtest
