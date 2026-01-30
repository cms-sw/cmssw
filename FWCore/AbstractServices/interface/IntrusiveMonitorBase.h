#ifndef FWCore_AbstractServices_IntrusiveMonitorBase_h
#define FWCore_AbstractServices_IntrusiveMonitorBase_h

#include <any>
#include <string>
#include <string_view>

namespace edm {
  /**
   * These objects form a linked list of nested uses
   * IntrusiveMonitors. The objects live in the stack via the Guard
   * object. The pointers to previous node are non-owning.
   *
   * This structure does not allocate memory
   */
  class IntrusiveMonitorStackNode {
  public:
    IntrusiveMonitorStackNode(std::string_view name, IntrusiveMonitorStackNode const* previous)
        : name_(name), previous_(previous) {}
    IntrusiveMonitorStackNode(IntrusiveMonitorStackNode const&) = delete;
    IntrusiveMonitorStackNode& operator=(IntrusiveMonitorStackNode const&) = delete;
    IntrusiveMonitorStackNode(IntrusiveMonitorStackNode&&) = delete;
    IntrusiveMonitorStackNode& operator=(IntrusiveMonitorStackNode&&) = delete;

    static void setNodeInCurrentThread(IntrusiveMonitorStackNode const* node);
    static IntrusiveMonitorStackNode const* getNodeInCurrentThread();

    std::string_view name() const { return name_; }
    IntrusiveMonitorStackNode const* previousNode() const { return previous_; }

  private:
    std::string_view name_;
    IntrusiveMonitorStackNode const* previous_;
  };

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
      Guard(IntrusiveMonitorBase& mon, T name)
          : monitor_(mon),
            name_(std::move(name)),
            value_(monitor_.start()),
            callStackNode_(name_, IntrusiveMonitorStackNode::getNodeInCurrentThread()) {
        IntrusiveMonitorStackNode::setNodeInCurrentThread(&callStackNode_);
      }
      Guard(Guard const&) = delete;
      Guard& operator=(Guard const&) = delete;
      Guard(Guard&&) = delete;
      Guard& operator=(Guard&&) = delete;

      ~Guard() {
        monitor_.stop(&callStackNode_, std::move(value_));
        IntrusiveMonitorStackNode::setNodeInCurrentThread(callStackNode_.previousNode());
      }

    private:
      IntrusiveMonitorBase& monitor_;
      T name_;
      std::any value_;
      IntrusiveMonitorStackNode callStackNode_;
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
    virtual std::any start() = 0;
    virtual void stop(IntrusiveMonitorStackNode const* callStack, std::any) = 0;
  };
}  // namespace edm

#endif
