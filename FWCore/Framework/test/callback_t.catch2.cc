/*
 *  callback_t.cc
 *  EDMProto
 *  Created by Chris Jones on 4/17/05.
 *  Changed by Viji Sundararajan on 03-Jul-05.
 *
 */

#include "catch2/catch_all.hpp"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Framework/interface/Callback.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

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
                 void const* ESParentContext) {}
    constexpr static bool allowConcurrentIOVs_ = false;
  };

  struct Queue {
    template <typename T>
    void push(oneapi::tbb::task_group&, T&& iT) {
      iT();
    }
  };

  struct Base {
    template <typename A, typename B>
    std::optional<std::vector<edm::ESResolverIndex>> updateFromMayConsumes(A const&, B const&) const {
      return {};
    }
    static constexpr edm::ESResolverIndex const* getTokenIndices(unsigned int) { return nullptr; }
    static constexpr edm::ESRecordIndex const* getTokenRecordIndices(unsigned int) { return nullptr; }
    static constexpr size_t numberOfTokenIndices(unsigned int) { return 0; }
    static constexpr bool hasMayConsumes() { return false; }
    static edm::eventsetup::ComponentDescription const& description() {
      static const edm::eventsetup::ComponentDescription s_description;
      return s_description;
    }

    Queue queue() { return Queue(); }
  };

  struct UniquePtrProd : public Base {
    constexpr UniquePtrProd() : value_(0) {}
    std::unique_ptr<Data> method(const Record&) {
      ++value_;
      if (produce_)
        return std::make_unique<Data>(value_);
      else
        return nullptr;
    }

    int value_;
    bool produce_ = true;
  };

  struct SharedPtrProd : public Base {
    SharedPtrProd() : ptr_(new Data()) {}
    std::shared_ptr<Data> method(const Record&) {
      ++ptr_->value_;
      if (produce_)
        return ptr_;
      else
        return nullptr;
    }
    std::shared_ptr<Data> ptr_;
    bool produce_ = true;
  };

  struct OptionalProd : public Base {
    constexpr OptionalProd() : value_(0) {}
    std::optional<Data> method(const Record&) {
      ++value_;
      if (produce_)
        return Data(value_);
      else
        return {};
    }

    int value_;
    bool produce_ = true;
  };

  struct PtrProductsProd : public Base {
    PtrProductsProd() : data_(), double_() {}
    edm::ESProducts<std::shared_ptr<Data>, std::shared_ptr<Double>> method(const Record&) {
      using namespace edm::es;
      auto dataT = std::shared_ptr<Data>(&data_, edm::do_nothing_deleter());
      auto doubleT = std::shared_ptr<Double>(&double_, edm::do_nothing_deleter());
      ++data_.value_;
      ++double_.value_;
      return products(dataT, doubleT);
    }

    Data data_;
    Double double_;
  };
}  // namespace callbacktest

EVENTSETUP_RECORD_REG(callbacktest::Record);

namespace {
  template <typename CALLBACK>
  void call(CALLBACK& iCallback) {
    edm::ActivityRegistry ar;
    edm::eventsetup::EventSetupRecordImpl rec(edm::eventsetup::EventSetupRecordKey::makeKey<callbacktest::Record>(),
                                              &ar);
    oneapi::tbb::task_group group;
    edm::FinalWaitingTask task{group};
    edm::ServiceToken token;
    iCallback.prefetchAsync(edm::WaitingTaskHolder(group, &task), &rec, nullptr, token, edm::ESParentContext{});
    task.wait();
  }
}  // namespace

using namespace callbacktest;
using namespace edm::eventsetup;

template <typename P, typename F>
using UniquePtrCallbackT = Callback<P, F, std::unique_ptr<Data>, Record>;

template <typename P, typename F>
using SharedPtrCallbackT = Callback<P, F, std::shared_ptr<Data>, Record>;

template <typename P, typename F>
using OptionalCallbackT = Callback<P, F, std::optional<Data>, Record>;

template <typename P, typename F>
using PtrProductsCallbackT = Callback<P, F, edm::ESProducts<std::shared_ptr<Data>, std::shared_ptr<Double>>, Record>;

TEST_CASE("Callback", "[Framework][EventSetup]") {
  auto m_scheduler = std::make_unique<edm::ThreadsController>(1);

  SECTION("uniquePtrTest") {
    UniquePtrProd prod;

    auto func = [&prod](Record const& rec) { return prod.method(rec); };
    using UniquePtrCallback = UniquePtrCallbackT<UniquePtrProd, decltype(func)>;
    UniquePtrCallback callback(&prod, func, 0);
    std::unique_ptr<Data> handle;
    callback.holdOntoPointer(&handle);

    auto callback2 = std::unique_ptr<UniquePtrCallback>(callback.clone());
    std::unique_ptr<Data> handle2;
    callback2->holdOntoPointer(&handle2);

    callback.newRecordComing();
    call(callback);
    REQUIRE(0 != handle.get());
    REQUIRE(prod.value_ == 1);
    assert(0 != handle.get());
    REQUIRE(prod.value_ == handle->value_);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(prod.value_ == 1);
    REQUIRE(prod.value_ == handle->value_);

    handle.release();

    callback.newRecordComing();

    call(callback);
    REQUIRE(0 != handle.get());
    REQUIRE(prod.value_ == 2);
    assert(0 != handle.get());
    REQUIRE(prod.value_ == handle->value_);

    call(*callback2);
    REQUIRE(handle2->value_ == 3);
    REQUIRE(handle->value_ == 2);

    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 3);
    REQUIRE(handle->value_ == 2);

    callback2->newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle->value_ == 2);

    callback.newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle->value_ == 5);

    // null products
    prod.produce_ = false;
    callback.newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle.get() == nullptr);

    callback2->newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2.get() == nullptr);
    REQUIRE(handle.get() == nullptr);

    prod.produce_ = true;
    callback.newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2.get() == nullptr);
    REQUIRE(handle->value_ == 8);

    callback2->newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 9);
    REQUIRE(handle->value_ == 8);
  }

  SECTION("sharedPtrTest") {
    SharedPtrProd prod;

    auto func = [&prod](Record const& rec) { return prod.method(rec); };

    using SharedPtrCallback = SharedPtrCallbackT<SharedPtrProd, decltype(func)>;
    SharedPtrCallback callback(&prod, func, 0);
    std::shared_ptr<Data> handle;

    callback.holdOntoPointer(&handle);

    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.get() == prod.ptr_.get());
    REQUIRE(prod.ptr_->value_ == 1);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle.get() == prod.ptr_.get());
    REQUIRE(prod.ptr_->value_ == 1);

    handle.reset();
    callback.newRecordComing();

    call(callback);
    REQUIRE(handle.get() == prod.ptr_.get());
    REQUIRE(prod.ptr_->value_ == 2);

    // null products
    prod.produce_ = false;
    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.get() == nullptr);

    call(callback);
    REQUIRE(handle.get() == nullptr);

    prod.produce_ = true;
    callback.newRecordComing();
    call(callback);
    REQUIRE(handle->value_ == 4);
  }

  SECTION("optionalTest") {
    OptionalProd prod;

    auto func = [&prod](Record const& rec) { return prod.method(rec); };

    using OptionalCallback = OptionalCallbackT<OptionalProd, decltype(func)>;
    OptionalCallback callback(&prod, func, 0);
    std::optional<Data> handle;

    callback.holdOntoPointer(&handle);

    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.has_value());
    REQUIRE(prod.value_ == 1);
    REQUIRE(prod.value_ == handle->value_);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle.has_value());
    REQUIRE(prod.value_ == 1);
    REQUIRE(prod.value_ == handle->value_);

    handle.reset();
    callback.newRecordComing();

    call(callback);
    REQUIRE(handle.has_value());
    REQUIRE(prod.value_ == 2);
    REQUIRE(prod.value_ == handle->value_);

    // null products
    prod.produce_ = false;
    callback.newRecordComing();
    call(callback);
    REQUIRE(not handle.has_value());

    call(callback);
    REQUIRE(not handle.has_value());

    prod.produce_ = true;
    callback.newRecordComing();
    call(callback);
    REQUIRE(handle->value_ == 4);
  }

  SECTION("ptrProductsTest") {
    PtrProductsProd prod;

    auto func = [&prod](Record const& rec) { return prod.method(rec); };

    using PtrProductsCallback = PtrProductsCallbackT<PtrProductsProd, decltype(func)>;
    PtrProductsCallback callback(&prod, func, 0);
    std::shared_ptr<Data> handle;
    std::shared_ptr<Double> doubleHandle;

    callback.holdOntoPointer(&handle);
    callback.holdOntoPointer(&doubleHandle);

    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.get() == &(prod.data_));
    REQUIRE(prod.data_.value_ == 1);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle.get() == &(prod.data_));
    REQUIRE(prod.data_.value_ == 1);

    callback.newRecordComing();

    call(callback);
    REQUIRE(handle.get() == &(prod.data_));
    REQUIRE(prod.data_.value_ == 2);
  }

  SECTION("uniquePtrLambdaTest") {
    Base prod;

    int value = 0;
    auto func = [&value](Record const& rec) mutable { return std::make_unique<Data>(++value); };
    using UniquePtrCallback = UniquePtrCallbackT<Base, decltype(func)>;
    UniquePtrCallback callback(&prod, func, 0);
    std::unique_ptr<Data> handle;
    callback.holdOntoPointer(&handle);

    auto callback2 = std::unique_ptr<UniquePtrCallback>(callback.clone());
    std::unique_ptr<Data> handle2;
    callback2->holdOntoPointer(&handle2);

    callback.newRecordComing();
    call(callback);
    REQUIRE(0 != handle.get());
    REQUIRE(value == 1);
    REQUIRE(value == handle->value_);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(value == 1);
    REQUIRE(value == handle->value_);

    handle.release();

    callback.newRecordComing();

    call(callback);
    REQUIRE(0 != handle.get());
    REQUIRE(value == 2);
    REQUIRE(value == handle->value_);

    call(*callback2);
    REQUIRE(handle2->value_ == 3);
    REQUIRE(handle->value_ == 2);

    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 3);
    REQUIRE(handle->value_ == 2);

    callback2->newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle->value_ == 2);

    callback.newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle->value_ == 5);
  }

  SECTION("uniquePtrLambdaCaptureTest") {
    // The difference wrt uniquePtrLambdaTest is that the 'value' is
    // stored only in the lambda capture
    Base prod;

    auto func = [value = int(0)](Record const& rec) mutable { return std::make_unique<Data>(++value); };
    using UniquePtrCallback = UniquePtrCallbackT<Base, decltype(func)>;
    UniquePtrCallback callback(&prod, func, 0);
    std::unique_ptr<Data> handle;
    callback.holdOntoPointer(&handle);

    auto callback2 = std::unique_ptr<UniquePtrCallback>(callback.clone());
    std::unique_ptr<Data> handle2;
    callback2->holdOntoPointer(&handle2);

    callback.newRecordComing();
    call(callback);
    REQUIRE(0 != handle.get());
    REQUIRE(handle->value_ == 1);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle->value_ == 1);

    handle.release();

    callback.newRecordComing();

    call(callback);
    REQUIRE(handle->value_ == 2);

    call(*callback2);
    REQUIRE(handle2->value_ == 3);
    REQUIRE(handle->value_ == 2);

    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 3);
    REQUIRE(handle->value_ == 2);

    callback2->newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle->value_ == 2);

    callback.newRecordComing();
    call(callback);
    call(*callback2);
    REQUIRE(handle2->value_ == 4);
    REQUIRE(handle->value_ == 5);
  }

  SECTION("sharedPtrLambdaTest") {
    Base prod;

    auto ptr = std::make_shared<Data>();
    auto func = [ptr](Record const& rec) mutable {
      ++ptr->value_;
      return ptr;
    };

    using SharedPtrCallback = SharedPtrCallbackT<Base, decltype(func)>;
    SharedPtrCallback callback(&prod, func, 0);
    std::shared_ptr<Data> handle;

    callback.holdOntoPointer(&handle);

    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.get() == ptr.get());
    REQUIRE(ptr->value_ == 1);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle.get() == ptr.get());
    REQUIRE(ptr->value_ == 1);

    handle.reset();
    callback.newRecordComing();

    call(callback);
    REQUIRE(handle.get() == ptr.get());
    REQUIRE(ptr->value_ == 2);
  }

  SECTION("optionalLambdaTest") {
    Base prod;

    int value = 0;
    auto func = [&value](Record const& rec) { return std::optional<Data>(++value); };

    using OptionalCallback = OptionalCallbackT<Base, decltype(func)>;
    OptionalCallback callback(&prod, func, 0);
    std::optional<Data> handle;

    callback.holdOntoPointer(&handle);

    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.has_value());
    REQUIRE(value == 1);
    REQUIRE(value == handle->value_);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle.has_value());
    REQUIRE(value == 1);
    REQUIRE(value == handle->value_);

    handle.reset();
    callback.newRecordComing();

    call(callback);
    REQUIRE(handle.has_value());
    REQUIRE(value == 2);
    REQUIRE(value == handle->value_);
  }

  SECTION("ptrProductsLambdaTest") {
    PtrProductsProd prod;

    Data dataValue;
    auto func = [&dataValue, doubleValue = Double()](Record const& rec) mutable {
      auto dataT = std::shared_ptr<Data>(&dataValue, edm::do_nothing_deleter());
      auto doubleT = std::shared_ptr<Double>(&doubleValue, edm::do_nothing_deleter());
      ++dataValue.value_;
      ++doubleValue.value_;
      return edm::es::products(dataT, doubleT);
    };

    using PtrProductsCallback = PtrProductsCallbackT<Base, decltype(func)>;
    PtrProductsCallback callback(&prod, func, 0);
    std::shared_ptr<Data> handle;
    std::shared_ptr<Double> doubleHandle;

    callback.holdOntoPointer(&handle);
    callback.holdOntoPointer(&doubleHandle);

    callback.newRecordComing();
    call(callback);
    REQUIRE(handle.get() == &dataValue);
    REQUIRE(dataValue.value_ == 1);

    //since haven't cleared, should not have changed
    call(callback);
    REQUIRE(handle.get() == &dataValue);
    REQUIRE(dataValue.value_ == 1);

    callback.newRecordComing();

    call(callback);
    REQUIRE(handle.get() == &dataValue);
    REQUIRE(dataValue.value_ == 2);
  }
}
