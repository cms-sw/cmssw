#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"

namespace {
  edm::IntrusiveMonitorStackNode const*& threadLocalPointer() {
    static thread_local edm::IntrusiveMonitorStackNode const* ptr = nullptr;
    return ptr;
  }
}  // namespace

namespace edm {
  void IntrusiveMonitorStackNode::setNodeInCurrentThread(IntrusiveMonitorStackNode const* node) {
    threadLocalPointer() = node;
  }

  IntrusiveMonitorStackNode const* IntrusiveMonitorStackNode::getNodeInCurrentThread() { return threadLocalPointer(); }

  IntrusiveMonitorBase::~IntrusiveMonitorBase() = default;
}  // namespace edm
