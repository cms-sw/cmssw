#ifndef PerfTools_AllocMonitor_ThreadTracker_h
#define PerfTools_AllocMonitor_ThreadTracker_h
// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     ThreadTracker
//
/**\class ThreadTracker ThreadTracker.h "ThreadTracker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 11 Nov 2024 22:54:21 GMT
//

// system include files
#if defined(ALLOC_USE_PTHREADS)
#include <pthread.h>
#else
#include <unistd.h>
#include <sys/syscall.h>
#endif

#include <array>
#include <atomic>
#include <cstdlib>
// user include files

// forward declarations

namespace cms::perftools::allocMon {
  inline auto thread_id() {
#if defined(ALLOC_USE_PTHREADS)
    /*NOTE: if use pthread_self, the values returned by linux had
     lots of hash collisions when using a simple % hash. Worked
     better if first divided value by 0x700 and then did %. 
     [test done on el8]
 */
    return pthread_self();
#else
    return syscall(SYS_gettid);
#endif
  }

  struct ThreadTracker {
    static constexpr unsigned int kHashedEntries = 128;
    static constexpr unsigned int kExtraEntries = 128;
    static constexpr unsigned int kTotalEntries = kHashedEntries + kExtraEntries;
    using entry_type = decltype(thread_id());
    static constexpr entry_type kUnusedEntry = ~entry_type(0);
    std::array<std::atomic<entry_type>, kHashedEntries> hashed_threads_;
    std::array<std::atomic<entry_type>, kExtraEntries> extra_threads_;

    std::size_t thread_index() {
      auto id = thread_id();
      auto index = thread_index_guess(id);
      auto used_id = hashed_threads_[index].load();

      if (id == used_id) {
        return index;
      }
      //try to be first thread to grab the index
      auto expected = entry_type(index + 1);
      if (used_id == expected) {
        if (hashed_threads_[index].compare_exchange_strong(expected, id)) {
          return index;
        } else {
          //another thread just beat us so have to go to non-hash storage
          return find_new_index(id);
        }
      }
      //search in non-hash storage
      return find_index(id);
    }

    static ThreadTracker& instance();

  private:
    ThreadTracker() {
      //put a value which will not match the % used when looking up the entry
      entry_type entry = 0;
      for (auto& v : hashed_threads_) {
        v = ++entry;
      }
      //assume kUsedEntry is not a valid thread-id
      for (auto& v : extra_threads_) {
        v = kUnusedEntry;
      }
    }

    std::size_t thread_index_guess(entry_type id) const {
#if defined(ALLOC_USE_PTHREADS)
      return (id / 0x700) % kHashedEntries;
#else
      return id % kHashedEntries;
#endif
    }

    std::size_t find_new_index(entry_type id) {
      std::size_t index = 0;
      for (auto& v : extra_threads_) {
        entry_type expected = kUnusedEntry;
        if (v == expected) {
          if (v.compare_exchange_strong(expected, id)) {
            return index + kHashedEntries;
          }
        }
        ++index;
      }
      //failed to find an open entry
      abort();
      return 0;
    }

    std::size_t find_index(entry_type id) {
      std::size_t index = 0;
      for (auto const& v : extra_threads_) {
        if (v == id) {
          return index + kHashedEntries;
        }
        ++index;
      }
      return find_new_index(id);
    }
  };
}  // namespace cms::perftools::allocMon
#endif
