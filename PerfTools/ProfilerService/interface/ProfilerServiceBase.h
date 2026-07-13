#ifndef FWCore_Services_ProfilerServiceBase_h__
#define FWCore_Services_ProfilerServiceBase_h__

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/container_hash/hash.hpp>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"

/// @brief Base class for profiling services.
/// @note This class contains the undelying utility classes.
class ProfilerServiceBase {
public:
  enum class Color : std::size_t {
    // Black, no variants
    Black = 0,
    // Red family (dark to light)
    Red_Dark2,
    Red_Dark1,
    Red,
    Red_Light1,
    Red_Light2,
    // Green family (dark to light)
    Green_Dark2,
    Green_Dark1,
    Green,
    Green_Light1,
    Green_Light2,
    // Blue family (dark to light)
    Blue_Dark2,
    Blue_Dark1,
    Blue,
    Blue_Light1,
    Blue_Light2,
    // Amber family (dark to light)
    Amber_Dark2,
    Amber_Dark1,
    Amber,
    Amber_Light1,
    Amber_Light2,
    // White, no variants
    White,
    // Grey family (dark to light)
    Grey_Dark2,
    Grey_Dark1,
    Grey,
    Grey_Light1,
    Grey_Light2,
    // Yellow family (dark to light)
    Yellow_Dark2,
    Yellow_Dark1,
    Yellow,
    Yellow_Light1,
    Yellow_Light2
  };

  static size_t to_underlying(Color c) noexcept { return static_cast<std::size_t>(c); }

  // Switch color to amber, but keep the same relative darkness/lightness level.
  static Color to_highlighted(Color c) noexcept {
    for (auto const start : {to_underlying(Color::Red_Dark2),
                             to_underlying(Color::Green_Dark2),
                             to_underlying(Color::Blue_Dark2),
                             to_underlying(Color::Amber_Dark2),
                             to_underlying(Color::Grey_Dark2),
                             to_underlying(Color::Yellow_Dark2)}) {
      auto const v = to_underlying(c);
      if (v >= start and v < start + 5)
        return static_cast<Color>(to_underlying(Color::Amber_Dark2) + (v - start));
    }
    return Color::Amber;  // singletons (Black / White)
  }

  /**
    * @brief Abstract color enumeration the derived classes can translate (or disregard).
    */
  class SpinLock {
  public:
    SpinLock() : flag_(ATOMIC_FLAG_INIT) {}

    void lock() {
      while (flag_.test_and_set(std::memory_order_acquire))
        ;
    }

    void unlock() { flag_.clear(std::memory_order_release); }

  private:
    std::atomic_flag flag_;
  };

  /// @brief Reader-writer spinlock.
  /// Compatible with std::lock_guard (exclusive) and std::shared_lock (shared).
  /// state_ == 0  : unlocked
  /// state_ == -1 : write-locked
  /// state_ >  0  : N concurrent readers
  class RWSpinLock {
  public:
    // Exclusive (write) access — use with std::lock_guard
    void lock() {
      int expected = 0;
      while (
          !state_.compare_exchange_weak(expected, kWriteLocked, std::memory_order_acquire, std::memory_order_relaxed)) {
        expected = 0;
      }
    }

    void unlock() { state_.store(0, std::memory_order_release); }

    // Shared (read) access — use with std::shared_lock
    void lock_shared() {
      while (true) {
        int val = state_.load(std::memory_order_relaxed);
        if (val >= 0 &&
            state_.compare_exchange_weak(val, val + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
          return;
        }
      }
    }

    void unlock_shared() { state_.fetch_sub(1, std::memory_order_release); }

  private:
    static constexpr int kWriteLocked = -1;
    std::atomic<int> state_{0};
  };

  template <typename Range>
  class RangePool {
  public:
    RangePool() : next_allocation_size_(kInitialAllocationSize) { allocateUnlocked_(kInitialAllocationSize); }

    size_t acquireSlot() {
      size_t slot = 0;
      bool got_slot = free_slots_.try_pop(slot);
      while (not got_slot) {
        std::lock_guard<SpinLock> guard(mutex_);
        allocateUnlocked_(next_allocation_size_);
        next_allocation_size_ *= 2;
        got_slot = free_slots_.try_pop(slot);
      }
      return slot;
    }

    void releaseSlot(size_t slot) { free_slots_.push(slot); }

    Range& at(size_t slot) { return ranges_[slot]; }

  private:
    static constexpr size_t kInitialAllocationSize = 16;

    void allocateUnlocked_(size_t count) {
      auto const begin = ranges_.size();
      ranges_.grow_by(count);
      for (size_t index = begin; index < begin + count; ++index) {
        free_slots_.push(index);
      }
    }

    SpinLock mutex_;
    size_t next_allocation_size_;
    tbb::concurrent_vector<Range> ranges_;
    tbb::concurrent_queue<size_t> free_slots_;
  };

  template <typename Backend, typename Range, typename Domain, typename... KeyArgs>
  class InFlightRanges {
  public:
    using Key = std::tuple<std::decay_t<KeyArgs>...>;

    explicit InFlightRanges(RangePool<Range>& range_pool, bool show_detailed_info = true)
        : range_pool_(range_pool), show_detailed_info_(show_detailed_info) {}

    // The range message is built automatically as the signal name followed by every key parameter
    // used for range indexing (see makeMessage_), so callers only pass the color, function, signal
    // and key arguments.
    void start(Domain& domain,
               Color color,
               char const* func,
               std::string_view signal,
               std::string_view detail,
               std::string_view keyNames,
               KeyArgs const&... keyArgs) {
      auto const msg = makeMessage_(signal, detail, keyNames, keyArgs...);
      auto const key = makeKey_(keyArgs...);
      auto const slot = range_pool_.acquireSlot();
      auto [found, inserted] = [&]() {
        std::shared_lock<RWSpinLock> guard(mutex_);
        return in_flight_.emplace(std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(slot));
      }();
      if (not inserted) {
        range_pool_.releaseSlot(slot);
        auto fullmsg = std::string("Warning: previous range not ended before starting a new one in ") + func +
                       " name=" + msg + " signal=" + std::string(signal);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      range_pool_.at(slot).startColorIn(domain, msg.c_str(), color, func);
    }

    void end(Domain& domain,
             char const* func,
             std::string_view signal,
             std::string_view keyNames,
             KeyArgs const&... keyArgs) {
      auto const key = makeKey_(keyArgs...);
      auto extracted = [&]() {
        std::lock_guard<RWSpinLock> guard(mutex_);
        return in_flight_.unsafe_extract(key);
      }();
      auto const msg = makeMessage_(signal, std::string_view{}, keyNames, keyArgs...);
      if (not extracted) {
        auto fullmsg = std::string("Warning: trying to end a range that is not started in ") + func + " name=" + msg +
                       " signal=" + std::string(signal);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      auto const slot = extracted.mapped();
      range_pool_.at(slot).endIn(domain, msg.c_str(), func);
      range_pool_.releaseSlot(slot);
    }

  private:
    static Key makeKey_(KeyArgs const&... keyArgs) { return Key{std::decay_t<KeyArgs>(keyArgs)...}; }

    // Stringify a single key argument: strings verbatim, enums via their underlying value, and any
    // other arithmetic type via std::to_string.
    template <typename T>
    static std::string keyToString_(T const& value) {
      using U = std::decay_t<T>;
      if constexpr (std::is_same_v<U, std::string>) {
        return value;
      } else if constexpr (std::is_enum_v<U>) {
        return std::to_string(static_cast<std::underlying_type_t<U>>(value));
      } else {
        return std::to_string(value);
      }
    }

    // Render the ES module calling state as a readable word rather than its numeric value.
    static std::string keyToString_(edm::ESModuleCallingContext::State state) {
      return state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching";
    }

    // Build the range message as the signal name, an optional human-readable detail (module label,
    // file name, path name, ...), and each indexing key parameter rendered as "<name>=<value>".
    // `keyNames` is a space-separated list matching the key arguments in order; an empty name emits
    // the bare value. A key that stringifies to the signal itself (e.g. the signal-string keyed
    // global ranges) is not repeated.
    std::string makeMessage_(std::string_view signal,
                             std::string_view detail,
                             std::string_view keyNames,
                             KeyArgs const&... keyArgs) {
      std::string msg{signal};
      if (not show_detailed_info_) {
        return msg;
      }
      if (not detail.empty()) {
        msg += ' ';
        msg += detail;
      }
      std::string_view rest = keyNames;
      auto nextName = [&]() -> std::string_view {
        auto const sp = rest.find(' ');
        auto const tok = rest.substr(0, sp);
        rest = (sp == std::string_view::npos) ? std::string_view{} : rest.substr(sp + 1);
        return tok;
      };
      auto append = [&](auto const& value) {
        auto const name = nextName();
        auto const s = keyToString_(value);
        if (s != signal) {
          msg += ' ';
          if (not name.empty()) {
            msg += name;
            msg += '=';
          }
          msg += s;
        }
      };
      (append(keyArgs), ...);
      return msg;
    }

    RWSpinLock mutex_;
    RangePool<Range>& range_pool_;
    bool show_detailed_info_;
    tbb::concurrent_unordered_map<Key, size_t, boost::hash<Key>> in_flight_;
  };
};

#endif  // FWCore_Services_ProfilerServiceBase_h__
