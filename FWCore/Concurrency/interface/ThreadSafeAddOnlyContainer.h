#ifndef FWCore_Concurrency_ThreadSafeAddOnlyContainer_h
#define FWCore_Concurrency_ThreadSafeAddOnlyContainer_h

#include <atomic>
#include <utility>

// This container will make objects and hold them. The container deletes
// the objects when the container is destroyed and only then. The client
// of the container should not delete them. The client must save the pointers
// to the objects that are returned by the makeAndHold function. There
// is no other way to access them. The pointers remain valid for as long
// as the container still exists.

// It is safe for multiple threads to concurrently call makeAndHold.

// Warning, none of the memory used by this is deallocated before the
// entire container is destroyed. If used in the wrong way, this container
// could cause memory hoarding.

// The original use case for this was that we had complex large objects
// in thread local storage and this was causing problems. Instead we
// we stored the complex objects in this container and used one thread
// local pointer to save the pointer to the object corresponding to
// to each thread. Instead of storing a complex object in thread local
// storage we were able to only store a simple pointer. There may be
// other uses for this.

namespace edm {

  template <typename T>
  class ThreadSafeAddOnlyContainer {
  public:
    ThreadSafeAddOnlyContainer();

    ~ThreadSafeAddOnlyContainer();

    template <typename... Args>
    T* makeAndHold(Args&&... args);

  private:
    class Node {
    public:
      template <typename... Args>
      Node(Node* iNext, Args&&... args);
      Node const* next() const { return next_; }
      void setNext(Node* v) { next_ = v; }
      T* address() { return &data_; }

    private:
      Node* next_;
      T data_;
    };

    std::atomic<Node*> front_;
  };

  template <typename T>
  ThreadSafeAddOnlyContainer<T>::ThreadSafeAddOnlyContainer() : front_(nullptr) {}

  template <typename T>
  ThreadSafeAddOnlyContainer<T>::~ThreadSafeAddOnlyContainer() {
    Node const* node = front_.load();
    while (node) {
      Node const* next = node->next();
      delete node;
      node = next;
    }
  }

  template <typename T>
  template <typename... Args>
  T* ThreadSafeAddOnlyContainer<T>::makeAndHold(Args&&... args) {
    Node* expected = front_.load();
    Node* newNode = new Node(expected, std::forward<Args>(args)...);
    while (!front_.compare_exchange_strong(expected, newNode)) {
      // another thread changed front_ before us so try again
      newNode->setNext(expected);
    }
    return newNode->address();
  }

  template <typename T>
  template <typename... Args>
  ThreadSafeAddOnlyContainer<T>::Node::Node(Node* iNext, Args&&... args)
      : next_(iNext), data_(std::forward<Args>(args)...) {}
}  // namespace edm

#endif
