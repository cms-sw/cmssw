#ifndef DQMServices_Core_ConcurrentMonitorElement_h
#define DQMServices_Core_ConcurrentMonitorElement_h

/* Encapsulate of MonitorElement to expose *limited* support for concurrency.
 *
 * ...
 */

#include <mutex>

#include "DQMServices/Core/interface/MonitorElement.h"

class ConcurrentMonitorElement : protected MonitorElement
{
private:
  mutable MonitorElement* me_;
  mutable std::mutex lock_;

public:
  ConcurrentMonitorElement(void) :
    me_(nullptr)
  { }

  explicit ConcurrentMonitorElement(MonitorElement* me) :
    me_(me)
  { }

  // non-copiable
  ConcurrentMonitorElement(ConcurrentMonitorElement const&) = delete;

  // movable
  ConcurrentMonitorElement(ConcurrentMonitorElement && other)
  {
    std::lock_guard<std::mutex>(other.lock_);
    me_ = other.me_;
    other.me_ = nullptr;
  }

  // not copy-assignable
  ConcurrentMonitorElement& operator=(ConcurrentMonitorElement const&) = delete;

  // move-assignable
  ConcurrentMonitorElement& operator=(ConcurrentMonitorElement && other)
  {
    // FIXME replace with std::scoped_lock once C++17 is available
    std::lock(lock_, other.lock_);
    std::lock_guard<std::mutex> ours(lock_, std::adopt_lock);
    std::lock_guard<std::mutex> others(other.lock_, std::adopt_lock);
    me_ = other.me_;
    other.me_ = nullptr;
    return *this;
  }

  // nothing to do, we do not own the MonitorElement
  ~ConcurrentMonitorElement(void) = default;

  // expose as a const method to mean that it is concurrent-safe
  template <typename... Args>
  void fill(Args && ... args) const
  {
    std::lock_guard<std::mutex> guard(lock_);
    me_->Fill(std::forward<Args>(args)...);
  }

  // expose as a const method to mean that it is concurrent-safe
  void shiftFillLast(double y, double ye = 0., int32_t xscale = 1) const
  {
    std::lock_guard<std::mutex> guard(lock_);
    me_->ShiftFillLast(y, ye, xscale);
  }

  // reset the internal pointer
  void reset()
  {
    std::lock_guard<std::mutex> guard(lock_);
    me_ = nullptr;
  }

  operator bool() const
  {
    std::lock_guard<std::mutex> guard(lock_);
    return (me_ != nullptr);
  } 
};

#endif // DQMServices_Core_ConcurrentMonitorElement_h
