#ifndef DQMServices_Core_ConcurrentMonitorElement_h
#define DQMServices_Core_ConcurrentMonitorElement_h

/* Encapsulate of MonitorElement to expose *limited* support for concurrency.
 *
 * ...
 */

#include <mutex>
#include <tbb/spin_mutex.h>

#include "DQMServices/Core/interface/MonitorElement.h"

class ConcurrentMonitorElement
{
private:
  mutable MonitorElement* me_;
  mutable tbb::spin_mutex lock_;

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
    std::lock_guard<tbb::spin_mutex> guard(other.lock_);
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
    std::lock_guard<tbb::spin_mutex> ours(lock_, std::adopt_lock);
    std::lock_guard<tbb::spin_mutex> others(other.lock_, std::adopt_lock);
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
    std::lock_guard<tbb::spin_mutex> guard(lock_);
    me_->Fill(std::forward<Args>(args)...);
  }

  // expose as a const method to mean that it is concurrent-safe
  void shiftFillLast(double y, double ye = 0., int32_t xscale = 1) const
  {
    std::lock_guard<tbb::spin_mutex> guard(lock_);
    me_->ShiftFillLast(y, ye, xscale);
  }

  // reset the internal pointer
  void reset()
  {
    std::lock_guard<tbb::spin_mutex> guard(lock_);
    me_ = nullptr;
  }

  operator bool() const
  {
    std::lock_guard<tbb::spin_mutex> guard(lock_);
    return (me_ != nullptr);
  }

  // non-const methods to manipulate axes and titles.
  // these are not concurrent-safe, and should be used only when the underlying
  // MonitorElement is being booked.
  void setTitle(std::string const& title)
  {
    me_->setTitle(title);
  }

  void setXTitle(std::string const& title)
  {
    me_->getTH1()->SetXTitle(title.c_str());
  }

  void setXTitle(const char* title)
  {
    me_->getTH1()->SetXTitle(title);
  }

  void setYTitle(std::string const& title)
  {
    me_->getTH1()->SetYTitle(title.c_str());
  }

  void setYTitle(const char* title)
  {
    me_->getTH1()->SetYTitle(title);
  }

  void setAxisRange(double xmin, double xmax, int axis = 1)
  {
    me_->setAxisRange(xmin, xmax, axis);
  }

  void setAxisTitle(std::string const& title, int axis = 1)
  {
    me_->setAxisTitle(title, axis);
  }

  void setAxisTimeDisplay(int value, int axis = 1)
  {
    me_->setAxisTimeDisplay(value, axis);
  }

  void setAxisTimeFormat(const char* format = "", int axis = 1)
  {
    me_->setAxisTimeFormat(format, axis);
  }

  void setBinLabel(int bin, std::string const& label, int axis = 1)
  {
    me_->setBinLabel(bin, label, axis);
  }

  void enableSumw2()
  {
    me_->getTH1()->Sumw2();
  }

  void disableAlphanumeric()
  {
    me_->getTH1()->GetXaxis()->SetNoAlphanumeric(false);
    me_->getTH1()->GetYaxis()->SetNoAlphanumeric(false);
  }

  void setOption(const char* option) {
    me_->getTH1()->SetOption(option);
  }
};

#endif // DQMServices_Core_ConcurrentMonitorElement_h
