#ifndef FWCore_AbstractServices_IntrusiveMonitorBase_h
#define FWCore_AbstractServices_IntrusiveMonitorBase_h

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

    class Guard {
    public:
      Guard(IntrusiveMonitorBase& mon, std::string_view name) : monitor_(mon), name_(name) { monitor_.start(); }
      Guard(Guard const&) = delete;
      Guard& operator=(Guard const&) = delete;
      Guard(Guard&&) = delete;
      Guard& operator=(Guard&&) = delete;

      ~Guard() { monitor_.stop(name_); }

    private:
      IntrusiveMonitorBase& monitor_;
      std::string_view name_;
    };

    Guard startMonitoring(std::string_view name) { return Guard(*this, name); }

  private:
    virtual void start() = 0;
    virtual void stop(std::string_view name) = 0;
  };
}  // namespace edm

#endif
