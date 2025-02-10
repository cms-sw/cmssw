#ifndef Geometry_CaloGeometry_CaloCellGeometryPtr_h
#define Geometry_CaloGeometry_CaloCellGeometryPtr_h
// -*- C++ -*-
//
// Package:     Geometry/CaloGeometry
// Class  :     CaloCellGeometryPtr
//
/**\class CaloCellGeometryPtr CaloCellGeometryPtr.h "Geometry/CaloGeometry/interface/CaloCellGeometryPtr.h"

 Description: Type to hold pointer to CaloCellGeometry

 Usage:
    Used to express that the CaloCellGeometry is owned elsewhere

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Oct 2024 18:47:22 GMT
//

// system include files

// user include files

// forward declarations
class CaloCellGeometry;

class CaloCellGeometryPtr {
public:
  explicit CaloCellGeometryPtr(CaloCellGeometry const* iPtr) noexcept : ptr_{iPtr} {}
  CaloCellGeometryPtr() noexcept = default;
  CaloCellGeometryPtr(const CaloCellGeometryPtr&) noexcept = default;
  CaloCellGeometryPtr(CaloCellGeometryPtr&&) noexcept = default;
  CaloCellGeometryPtr& operator=(CaloCellGeometryPtr const&) noexcept = default;
  CaloCellGeometryPtr& operator=(CaloCellGeometryPtr&&) noexcept = default;

  CaloCellGeometry const* operator->() const { return ptr_; }
  CaloCellGeometry const* get() const { return ptr_; }
  CaloCellGeometry const& operator*() const { return *ptr_; }

  operator CaloCellGeometry const*() const { return ptr_; }

private:
  const CaloCellGeometry* ptr_ = nullptr;
};

#endif
