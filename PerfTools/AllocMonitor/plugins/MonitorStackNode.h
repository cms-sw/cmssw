#ifndef PerfTools_AllocMonitor_plugins_MonitorStackNode_h
#define PerfTools_AllocMonitor_plugins_MonitorStackNode_h

#include <memory>

namespace cms::perftools::allocMon {
  /**
   * This class template provides a common stack node structure for
   * IntrusiveAllocMonitors. The stack is presented as a linked list
   * of nested uses of thean IntrusiveAllocMonitor measurements.
   */
  template <typename T>
  class MonitorStackNode {
  public:
    MonitorStackNode(std::string_view name, std::unique_ptr<MonitorStackNode> previousNode, T&& data)
        : name_(name), previousNode_(std::move(previousNode)), data_(std::move(data)) {}

    MonitorStackNode(MonitorStackNode const&) = delete;
    MonitorStackNode& operator=(MonitorStackNode const&) = delete;
    MonitorStackNode(MonitorStackNode&&) = delete;
    MonitorStackNode& operator=(MonitorStackNode&&) = delete;
    ~MonitorStackNode() noexcept = default;

    std::string_view name() const { return name_; }
    MonitorStackNode* previousNode() { return previousNode_.get(); }
    MonitorStackNode const* previousNode() const { return previousNode_.get(); }
    std::unique_ptr<MonitorStackNode> popPreviousNode() { return std::move(previousNode_); }

    T& get() { return data_; }
    T const& get() const { return data_; }

  private:
    std::string_view name_;
    std::unique_ptr<MonitorStackNode> previousNode_;
    T data_;
  };
}  // namespace cms::perftools::allocMon

#endif
