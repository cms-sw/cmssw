#ifndef FWCore_Utilities_ReusableObjectHolder_h
#define FWCore_Utilities_ReusableObjectHolder_h

// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     ReusableObjectHolder
//
/**\class edm::ReusableObjectHolder ReusableObjectHolder "ReusableObjectHolder.h"
 
 Description: Thread safe way to do create and reuse a group of the same object type.
 
 Usage:
 This class can be used to safely reuse a series of objects created on demand. The reuse
 of the objects is safe even across different threads since one can safely call all member
 functions of this class on the same instance of this class from multiple threads.

 This class manages the cache of reusable objects and therefore an instance of this
 class must live as long as you want the cache to live.

 The primary way of using the class is to call makeOrGet:
 \code
 auto objectToUse = holder.makeOrGet([]() { return new MyObject(); });
 use(*objectToUse);
 \endcode

 If the returned object should be be automatically set or reset, call makeOrGetAndClear:
 \code
 auto objectToUse = holder.makeOrGetAndClear([]() { return new MyObject(10); },   // makes new objects
                                             [](MyObject* old) { old->reset(); }  // resets any object before returning it
 );
 \endcode
 which is equivalent to
 \code
 auto objectToUse = holder.makeOrGet([]() { return new MyObject(10); });
 objectToUse->reset();
 \endcode
 
 NOTE: If you hold onto the std::shared_ptr<> until another call to the ReusableObjectHolder,
 make sure to release the shared_ptr before the call. That way the object you were just
 using can go back into the cache and be reused for the call you are going to make.
 An example
 \code
  std::shared_ptr<MyObject> obj;
  while(someCondition()) {
    //release object so it can re-enter the cache
    obj.release();
    obj = holder.makeOrGet([]{ return new MyObject();} );
    obj->setValue(someNewValue());
    useTheObject(obj);
  }
 \endcode
 
 The above example is very contrived, since the better way to do the above is
 \code
 while(someCondition()) {
   auto obj = holder.makeOrGet([]{ return new MyObject();} );
   obj->setValue(someNewValue());
   useTheObject(obj);
   //obj goes out of scope and returns the object to the cache
 }
 \endcode

 When a custom deleter is used, the deleter type must be the same to
 all objects. The deleter is allowed to have state that depends on the
 object. The deleter object is passed along the std::unique_ptr, and
 is internally kept along the object. The deleter object must be copyable.
 */
//
// Original Author:  Chris Jones
//         Created:  Fri, 31 July 2014 14:29:41 GMT
//

#include <atomic>
#include <cassert>
#include <memory>

#include <oneapi/tbb/concurrent_queue.h>

namespace edm {
  template <class T, class Deleter = std::default_delete<T>>
  class ReusableObjectHolder {
  public:
    using deleter_type = Deleter;

    ReusableObjectHolder() : m_outstandingObjects(0) {}
    ReusableObjectHolder(ReusableObjectHolder&& iOther)
        : m_availableQueue(std::move(iOther.m_availableQueue)), m_outstandingObjects(0) {
      assert(0 == iOther.m_outstandingObjects);
    }
    ~ReusableObjectHolder() {
      assert(0 == m_outstandingObjects);
      std::unique_ptr<T, Deleter> item;
      while (m_availableQueue.try_pop(item)) {
        item.reset();
      }
    }

    ///Adds the item to the cache.
    /// Use this function if you know ahead of time
    /// how many cached items you will need.
    void add(std::unique_ptr<T, Deleter> iItem) {
      if (nullptr != iItem) {
        m_availableQueue.push(std::move(iItem));
      }
    }

    ///Tries to get an already created object,
    /// if none are available, returns an empty shared_ptr.
    /// Use this function in conjunction with add()
    std::shared_ptr<T> tryToGet() {
      std::unique_ptr<T, Deleter> item;
      if (m_availableQueue.try_pop(item)) {
        return wrapCustomDeleter(std::move(item));
      } else {
        return std::shared_ptr<T>{};
      }
    }

    ///Takes an object from the queue if one is available, or creates one using iMakeFunc.
    template <typename FM>
    std::shared_ptr<T> makeOrGet(FM&& iMakeFunc) {
      std::unique_ptr<T, Deleter> item;
      if (m_availableQueue.try_pop(item)) {
        return wrapCustomDeleter(std::move(item));
      } else {
        return wrapCustomDeleter(makeUnique(iMakeFunc()));
      }
    }

    ///Takes an object from the queue if one is available, or creates one using iMakeFunc.
    ///Then, passes the object to iClearFunc, and returns it.
    template <typename FM, typename FC>
    std::shared_ptr<T> makeOrGetAndClear(FM&& iMakeFunc, FC&& iClearFunc) {
      std::shared_ptr<T> returnValue = makeOrGet(std::forward<FM>(iMakeFunc));
      iClearFunc(returnValue.get());
      return returnValue;
    }

  private:
    ///Wraps an object in a shared_ptr<T> with a custom deleter, that hands the wrapped object
    // back to the queue instead of deleting it
    std::shared_ptr<T> wrapCustomDeleter(std::unique_ptr<T, Deleter> item) {
      auto deleter = item.get_deleter();
      ++m_outstandingObjects;
      return std::shared_ptr<T>{item.release(), [this, deleter](T* iItem) {
                                  this->addBack(std::unique_ptr<T, Deleter>{iItem, deleter});
                                }};
    }

    std::unique_ptr<T> makeUnique(T* ptr) {
      static_assert(std::is_same_v<Deleter, std::default_delete<T>>,
                    "Generating functions returning raw pointers are supported only with std::default_delete<T>");
      return std::unique_ptr<T>{ptr};
    }

    std::unique_ptr<T, Deleter> makeUnique(std::unique_ptr<T, Deleter> ptr) { return ptr; }

    void addBack(std::unique_ptr<T, Deleter> iItem) {
      m_availableQueue.push(std::move(iItem));
      --m_outstandingObjects;
    }

    oneapi::tbb::concurrent_queue<std::unique_ptr<T, Deleter>> m_availableQueue;
    std::atomic<size_t> m_outstandingObjects;
  };

}  // namespace edm

#endif /* end of include guard: FWCore_Utilities_ReusableObjectHolder_h */
