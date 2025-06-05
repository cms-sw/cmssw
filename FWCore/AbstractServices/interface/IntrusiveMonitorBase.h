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
    virtual ~IntrusiveMonitorBase();

    template <typename T>
      requires std::is_same_v<T, std::string> or std::is_same_v<T, std::string_view>
    class Guard {
    public:
      Guard(IntrusiveMonitorBase& mon, T name) : monitor_(mon), name_(std::move(name)) { monitor_.start(); }
      Guard(Guard const&) = delete;
      Guard& operator=(Guard const&) = delete;
      Guard(Guard&&) = delete;
      Guard& operator=(Guard&&) = delete;

      ~Guard() { monitor_.stop(name_); }

    private:
      IntrusiveMonitorBase& monitor_;
      T name_;
    };

    auto startMonitoring(std::string_view name) { return Guard<std::string_view>(*this, name); }

    // direct std::string&& would be ambiguous for C-string literals
    // without is_rvalue_reference this overload would match for lvalue std::string
    template <typename T>
      requires std::is_same_v<std::remove_cvref_t<T>, std::string> and std::is_rvalue_reference_v<T&&>
    auto startMonitoring(T&& name) {
      return Guard<std::string>(*this, std::move(name));
    }

  private:
    virtual void start() = 0;
    virtual void stop(std::string_view name) = 0;
  };
}  // namespace edm

#endif
