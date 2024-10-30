#ifndef Geometry_CaloGeometry_CaloCellGeometryMayOwnPtr_h
#define Geometry_CaloGeometry_CaloCellGeometryMayOwnPtr_h
// -*- C++ -*-
//
// Package:     Geometry/CaloGeometry
// Class  :     CaloCellGeometryMayOwnPtr
//
/**\class CaloCellGeometryMayOwnPtr CaloCellGeometryMayOwnPtr.h "Geometry/CaloGeometry/interface/CaloCellGeometryMayOwnPtr.h"

 Description: Type to hold pointer to CaloCellGeometry with possible ownership

 Usage:
    Used to either have single ownership or no ownership of the CaloCellGeometry

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Oct 2024 19:50:05 GMT
//

// system include files
#include <memory>

// user include files
#include "Geometry/CaloGeometry/interface/CaloCellGeometryPtr.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

// forward declarations

class CaloCellGeometryMayOwnPtr {
public:
  explicit CaloCellGeometryMayOwnPtr(std::unique_ptr<CaloCellGeometry const> iPtr) noexcept
      : ptr_{iPtr.release()}, own_{ptr_ != nullptr} {
    if (own_) {
      ptr_->increment();
    }
  }
  explicit CaloCellGeometryMayOwnPtr(CaloCellGeometryPtr const& iPtr) noexcept : ptr_{iPtr.get()}, own_{false} {}

  ~CaloCellGeometryMayOwnPtr() noexcept {
    if (own_ and ptr_->decrement()) {
      delete ptr_;
    }
  }
  CaloCellGeometryMayOwnPtr() noexcept = default;
  CaloCellGeometryMayOwnPtr(const CaloCellGeometryMayOwnPtr& iPtr) noexcept : ptr_{iPtr.ptr_}, own_{iPtr.own_} {
    if (own_) {
      ptr_->increment();
    }
  }
  CaloCellGeometryMayOwnPtr(CaloCellGeometryMayOwnPtr&& iPtr) noexcept : ptr_{iPtr.ptr_}, own_{iPtr.own_} {
    iPtr.ptr_ = nullptr;
    iPtr.own_ = false;
  }
  CaloCellGeometryMayOwnPtr& operator=(CaloCellGeometryMayOwnPtr const& iPtr) noexcept {
    //Even if someone does `foo = foo` this will work
    auto tmpPtr = iPtr.ptr_;
    auto tmpOwn = iPtr.own_;
    CaloCellGeometryMayOwnPtr temp(std::move(*this));
    ptr_ = tmpPtr;
    own_ = tmpOwn;
    if (own_) {
      ptr_->increment();
    }
    return *this;
  }
  CaloCellGeometryMayOwnPtr& operator=(CaloCellGeometryMayOwnPtr&& iPtr) noexcept {
    if (&iPtr != this) {
      CaloCellGeometryMayOwnPtr temp(std::move(*this));

      ptr_ = iPtr.ptr_;
      own_ = iPtr.own_;
      iPtr.ptr_ = nullptr;
      iPtr.own_ = false;
    }
    return *this;
  }

  CaloCellGeometry const* operator->() const { return ptr_; }
  CaloCellGeometry const* get() const { return ptr_; }
  CaloCellGeometry const& operator*() const { return *ptr_; }

  operator CaloCellGeometry const*() const { return ptr_; }

private:
  struct no_delete {
    void operator()(const void*) {}
  };

  const CaloCellGeometry* ptr_ = nullptr;
  bool own_ = false;
};

#endif
