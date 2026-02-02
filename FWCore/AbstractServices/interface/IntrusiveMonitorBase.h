#ifndef FWCore_AbstractServices_IntrusiveMonitorBase_h
#define FWCore_AbstractServices_IntrusiveMonitorBase_h

#include <string>
#include <string_view>

namespace edm {
  class IntrusiveMonitorBase {
  public:
    IntrusiveMonitorBase() = default;
    IntrusiveMonitorBase(IntrusiveMonitorBase const&) = delete;
    IntrusiveMonitorBase& operator=(IntrusiveMonitorBase const&) = delete;
    IntrusiveMonitorBase(IntrusiveMonitorBase&&) = delete;
    IntrusiveMonitorBase& operator=(IntrusiveMonitorBase&&) = delete;
    virtual ~IntrusiveMonitorBase() noexcept;

    template <typename T>
    class Guard;

    Guard<std::string_view> startMonitoring(std::string_view name);

    // direct std::string&& would be ambiguous for C-string literals
    // without is_rvalue_reference this overload would match for lvalue std::string
    template <typename T>
      requires std::is_same_v<std::remove_cvref_t<T>, std::string> and std::is_rvalue_reference_v<T&&>
    Guard<std::string> startMonitoring(T&& name);

  private:
    virtual void start(std::string_view name, bool nameIsString) = 0;
    virtual void stop() noexcept = 0;
  };

  template <>
  class IntrusiveMonitorBase::Guard<std::string> {
  public:
    Guard(IntrusiveMonitorBase& mon, std::string name) : monitor_(mon), name_(std::move(name)) {
      monitor_.start(name_, true);
    }
    Guard(Guard const&) = delete;
    Guard& operator=(Guard const&) = delete;
    Guard(Guard&&) = delete;
    Guard& operator=(Guard&&) = delete;

    ~Guard() noexcept { monitor_.stop(); }

  private:
    IntrusiveMonitorBase& monitor_;
    // The std::string must be kept alive until the monitor_.stop() call finishes
    std::string name_;
  };

  template <>
  class IntrusiveMonitorBase::Guard<std::string_view> {
  public:
    Guard(IntrusiveMonitorBase& mon, std::string_view name) : monitor_(mon) { monitor_.start(name, false); }
    Guard(Guard const&) = delete;
    Guard& operator=(Guard const&) = delete;
    Guard(Guard&&) = delete;
    Guard& operator=(Guard&&) = delete;

    ~Guard() noexcept { monitor_.stop(); }

  private:
    IntrusiveMonitorBase& monitor_;
  };

  inline IntrusiveMonitorBase::Guard<std::string_view> IntrusiveMonitorBase::startMonitoring(std::string_view name) {
    return Guard<std::string_view>(*this, name);
  }

  template <typename T>
    requires std::is_same_v<std::remove_cvref_t<T>, std::string> and std::is_rvalue_reference_v<T&&>
  IntrusiveMonitorBase::Guard<std::string> IntrusiveMonitorBase::startMonitoring(T&& name) {
    return Guard<std::string>(*this, std::move(name));
  }

}  // namespace edm

#endif
