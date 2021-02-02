/*
 *  callback_t.cc
 *  EDMProto
 *  Created by Chris Jones on 4/17/05.
 *  Changed by Viji Sundararajan on 03-Jul-05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Framework/interface/Callback.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include <memory>
#include <cassert>

namespace callbacktest {
  struct Data {
    Data() : value_(0) {}
    Data(int iValue) : value_(iValue) {}
    virtual ~Data() {}
    int value_;
  };

  struct Double {
    Double() : value_(0) {}
    Double(double iValue) : value_(iValue) {}
    virtual ~Double() {}
    double value_;
  };

  struct Record {
    void setImpl(void const* iImpl,
                 unsigned int transitionID,
                 void const* getTokenIndices,
                 void const* iEventSetupImpl,
                 bool requireTokens) {}
  };

  struct Queue {
    template <typename T>
    void push(T&& iT) {
      iT();
    }
  };

  struct ComponentDescription {
    static constexpr char const* const type_ = "";
    static constexpr char const* const label_ = "";
  };

  struct Base {
    template <typename A, typename B>
    std::optional<std::vector<edm::ESProxyIndex>> updateFromMayConsumes(A const&, B const&) const {
      return {};
    }
    static constexpr edm::ESProxyIndex const* getTokenIndices(unsigned int) { return nullptr; }
    static constexpr edm::ESRecordIndex const* getTokenRecordIndices(unsigned int) { return nullptr; }
    static constexpr size_t numberOfTokenIndices(unsigned int) { return 0; }
    static constexpr bool hasMayConsumes() { return false; }
    static ComponentDescription description() { return ComponentDescription{}; }

    Queue queue() { return Queue(); }
  };

  struct UniquePtrProd : public Base {
    constexpr UniquePtrProd() : value_(0) {}
    std::unique_ptr<Data> method(const Record&) { return std::make_unique<Data>(++value_); }

    int value_;
  };

  struct SharedPtrProd : public Base {
    SharedPtrProd() : ptr_(new Data()) {}
    std::shared_ptr<Data> method(const Record&) {
      ++ptr_->value_;
      return ptr_;
    }
    std::shared_ptr<Data> ptr_;
  };

  struct PtrProductsProd : public Base {
    PtrProductsProd() : data_(), double_() {}
    edm::ESProducts<std::shared_ptr<Data>, std::shared_ptr<Double>> method(const Record&) {
      using namespace edm::es;
      auto dataT = std::shared_ptr<Data>(&data_, edm::do_nothing_deleter());
      ;
      auto doubleT = std::shared_ptr<Double>(&double_, edm::do_nothing_deleter());
      ++data_.value_;
      ++double_.value_;
      return products(dataT, doubleT);
    }

    Data data_;
    Double double_;
  };
}  // namespace callbacktest

namespace {
  template <typename CALLBACK>
  void call(CALLBACK& iCallback) {
    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(1);
    iCallback.prefetchAsync(edm::WaitingTaskHolder(waitTask.get()), nullptr, nullptr, edm::ServiceToken{});
    waitTask->wait_for_all();
  }
}  // namespace

using namespace callbacktest;
using namespace edm::eventsetup;

class testCallback : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCallback);

  CPPUNIT_TEST(uniquePtrTest);
  CPPUNIT_TEST(sharedPtrTest);
  CPPUNIT_TEST(ptrProductsTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<edm::ThreadsController>(1); }
  void tearDown() {}

  void uniquePtrTest();
  void sharedPtrTest();
  void ptrProductsTest();

private:
  edm::propagate_const<std::unique_ptr<edm::ThreadsController>> m_scheduler;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCallback);

using UniquePtrCallback = Callback<UniquePtrProd, std::unique_ptr<Data>, Record>;

void testCallback::uniquePtrTest() {
  UniquePtrProd prod;

  UniquePtrCallback callback(&prod, &UniquePtrProd::method, 0);
  std::unique_ptr<Data> handle;
  callback.holdOntoPointer(&handle);

  auto callback2 = std::unique_ptr<UniquePtrCallback>(callback.clone());
  std::unique_ptr<Data> handle2;
  callback2->holdOntoPointer(&handle2);

  callback.newRecordComing();
  call(callback);
  CPPUNIT_ASSERT(0 != handle.get());
  CPPUNIT_ASSERT(prod.value_ == 1);
  assert(0 != handle.get());
  CPPUNIT_ASSERT(prod.value_ == handle->value_);

  //since haven't cleared, should not have changed
  call(callback);
  CPPUNIT_ASSERT(prod.value_ == 1);
  CPPUNIT_ASSERT(prod.value_ == handle->value_);

  handle.release();

  callback.newRecordComing();

  call(callback);
  CPPUNIT_ASSERT(0 != handle.get());
  CPPUNIT_ASSERT(prod.value_ == 2);
  assert(0 != handle.get());
  CPPUNIT_ASSERT(prod.value_ == handle->value_);

  call(*callback2);
  CPPUNIT_ASSERT(handle2->value_ == 3);
  CPPUNIT_ASSERT(handle->value_ == 2);

  call(callback);
  call(*callback2);
  CPPUNIT_ASSERT(handle2->value_ == 3);
  CPPUNIT_ASSERT(handle->value_ == 2);

  callback2->newRecordComing();
  call(callback);
  call(*callback2);
  CPPUNIT_ASSERT(handle2->value_ == 4);
  CPPUNIT_ASSERT(handle->value_ == 2);
}

typedef Callback<SharedPtrProd, std::shared_ptr<Data>, Record> SharedPtrCallback;

void testCallback::sharedPtrTest() {
  SharedPtrProd prod;

  SharedPtrCallback callback(&prod, &SharedPtrProd::method, 0);
  std::shared_ptr<Data> handle;

  callback.holdOntoPointer(&handle);

  callback.newRecordComing();
  call(callback);
  CPPUNIT_ASSERT(handle.get() == prod.ptr_.get());
  CPPUNIT_ASSERT(prod.ptr_->value_ == 1);

  //since haven't cleared, should not have changed
  call(callback);
  CPPUNIT_ASSERT(handle.get() == prod.ptr_.get());
  CPPUNIT_ASSERT(prod.ptr_->value_ == 1);

  handle.reset();
  callback.newRecordComing();

  call(callback);
  CPPUNIT_ASSERT(handle.get() == prod.ptr_.get());
  CPPUNIT_ASSERT(prod.ptr_->value_ == 2);
}

typedef Callback<PtrProductsProd, edm::ESProducts<std::shared_ptr<Data>, std::shared_ptr<Double>>, Record>
    PtrProductsCallback;

void testCallback::ptrProductsTest() {
  PtrProductsProd prod;

  PtrProductsCallback callback(&prod, &PtrProductsProd::method, 0);
  std::shared_ptr<Data> handle;
  std::shared_ptr<Double> doubleHandle;

  callback.holdOntoPointer(&handle);
  callback.holdOntoPointer(&doubleHandle);

  callback.newRecordComing();
  call(callback);
  CPPUNIT_ASSERT(handle.get() == &(prod.data_));
  CPPUNIT_ASSERT(prod.data_.value_ == 1);

  //since haven't cleared, should not have changed
  call(callback);
  CPPUNIT_ASSERT(handle.get() == &(prod.data_));
  CPPUNIT_ASSERT(prod.data_.value_ == 1);

  callback.newRecordComing();

  call(callback);
  CPPUNIT_ASSERT(handle.get() == &(prod.data_));
  CPPUNIT_ASSERT(prod.data_.value_ == 2);
}
