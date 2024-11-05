// -*- C++ -*-
//
// Package:     Geometry/CaloGeometry
// Class  :     calocellgeometryptr_catch2
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 24 Oct 2024 17:35:52 GMT
//

// system include files

// user include files
#include "Geometry/CaloGeometry/interface/CaloCellGeometryPtr.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometryMayOwnPtr.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace {
  class DummyCell : public CaloCellGeometry {
  public:
    DummyCell() { ++nLive; }
    ~DummyCell() { --nLive; }
    DummyCell(DummyCell const&) = delete;
    DummyCell(DummyCell&&) = delete;
    DummyCell& operator=(DummyCell const&) = delete;
    DummyCell& operator=(DummyCell&&) = delete;

    void vocalCorners(Pt3DVec& vec, const CCGFloat* pv, Pt3D& ref) const final {}
    void initCorners(CornersVec&) final {}

    static int nLive;
  };
  int DummyCell::nLive = 0;
}  // namespace

TEST_CASE("Test CaloCellGeometryPtr", "[CaloCellGeometryPtr]") {
  SECTION("Default Constructor") {
    CaloCellGeometryPtr ptr;
    CHECK(ptr.get() == nullptr);
    CHECK(static_cast<CaloCellGeometry const*>(ptr) == nullptr);
  }
  SECTION("Pointer Constructor") {
    {
      DummyCell dummy;
      REQUIRE(DummyCell::nLive == 1);

      {
        CaloCellGeometryPtr ptr(&dummy);
        CHECK(ptr.get() == &dummy);
        CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
        CHECK(ptr.operator->() == &dummy);
        CHECK(&(*ptr) == &dummy);
      }
      REQUIRE(DummyCell::nLive == 1);
    }
    REQUIRE(DummyCell::nLive == 0);
  }
}

TEST_CASE("Test CaloCellGeometryMayOwnPtr", "[CaloCellGeometryPtr]") {
  SECTION("Default Constructed") {
    CaloCellGeometryMayOwnPtr ptr;
    CHECK(ptr.get() == nullptr);
    CHECK(static_cast<CaloCellGeometry const*>(ptr) == nullptr);
  }
  SECTION("From CaloCellGeometryPtr") {
    {
      DummyCell dummy;
      REQUIRE(DummyCell::nLive == 1);
      CaloCellGeometryPtr p(&dummy);
      {
        CaloCellGeometryMayOwnPtr ptr(p);
        CHECK(ptr.get() == &dummy);
        CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
        CHECK(ptr.operator->() == &dummy);
        CHECK(&(*ptr) == &dummy);
      }
      REQUIRE(DummyCell::nLive == 1);
    }
    REQUIRE(DummyCell::nLive == 0);
  }

  SECTION("From unique_ptr") {
    {
      auto dummy = std::make_unique<DummyCell>();
      auto dummyAddress = dummy.get();
      REQUIRE(DummyCell::nLive == 1);
      {
        CaloCellGeometryMayOwnPtr ptr(std::move(dummy));
        CHECK(ptr.get() == dummyAddress);
        CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
        CHECK(ptr.operator->() == dummyAddress);
        CHECK(&(*ptr) == dummyAddress);
      }
      REQUIRE(DummyCell::nLive == 0);
    }
  }

  SECTION("move constructor") {
    SECTION("non-owning") {
      DummyCell dummy;
      REQUIRE(DummyCell::nLive == 1);
      CaloCellGeometryPtr p(&dummy);
      {
        CaloCellGeometryMayOwnPtr from(p);
        {
          CaloCellGeometryMayOwnPtr ptr(std::move(from));
          CHECK(from.get() == nullptr);
          CHECK(ptr.get() == &dummy);
          CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
          CHECK(ptr.operator->() == &dummy);
          CHECK(&(*ptr) == &dummy);
          REQUIRE(DummyCell::nLive == 1);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
      REQUIRE(DummyCell::nLive == 1);
    }
    SECTION("owning") {
      auto dummy = std::make_unique<DummyCell>();
      auto dummyAddress = dummy.get();
      REQUIRE(DummyCell::nLive == 1);
      {
        CaloCellGeometryMayOwnPtr from(std::move(dummy));
        {
          CaloCellGeometryMayOwnPtr ptr(std::move(from));
          CHECK(from.get() == nullptr);
          CHECK(ptr.get() == dummyAddress);
          CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
          CHECK(ptr.operator->() == dummyAddress);
          CHECK(&(*ptr) == dummyAddress);
        }
        REQUIRE(DummyCell::nLive == 0);
      }
    }
  }

  SECTION("copy constructor") {
    SECTION("non-owning") {
      DummyCell dummy;
      REQUIRE(DummyCell::nLive == 1);
      {
        CaloCellGeometryPtr p(&dummy);
        {
          CaloCellGeometryMayOwnPtr from(p);
          {
            CaloCellGeometryMayOwnPtr ptr(from);
            CHECK(from.get() == &dummy);
            CHECK(ptr.get() == &dummy);
            CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
            CHECK(ptr.operator->() == &dummy);
            CHECK(&(*ptr) == &dummy);
            REQUIRE(DummyCell::nLive == 1);
          }
          REQUIRE(DummyCell::nLive == 1);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
      REQUIRE(DummyCell::nLive == 1);
    }
    SECTION("owning") {
      auto dummy = std::make_unique<DummyCell>();
      auto dummyAddress = dummy.get();
      REQUIRE(DummyCell::nLive == 1);
      {
        CaloCellGeometryMayOwnPtr from(std::move(dummy));
        {
          CaloCellGeometryMayOwnPtr ptr(from);
          CHECK(from.get() == dummyAddress);
          CHECK(ptr.get() == dummyAddress);
          CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
          CHECK(ptr.operator->() == dummyAddress);
          CHECK(&(*ptr) == dummyAddress);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
      REQUIRE(DummyCell::nLive == 0);
    }
  }

  SECTION("move assignment") {
    SECTION("from non-owning") {
      DummyCell oldDummy;
      CaloCellGeometryPtr p(&oldDummy);
      REQUIRE(DummyCell::nLive == 1);
      SECTION("to non-owning") {
        DummyCell dummy;
        REQUIRE(DummyCell::nLive == 2);
        {
          CaloCellGeometryMayOwnPtr ptr(p);
          CaloCellGeometryPtr p(&dummy);
          {
            CaloCellGeometryMayOwnPtr from(p);
            {
              ptr = std::move(from);
              CHECK(from.get() == nullptr);
              CHECK(ptr.get() == &dummy);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
              CHECK(ptr.operator->() == &dummy);
              CHECK(&(*ptr) == &dummy);
            }
            REQUIRE(DummyCell::nLive == 2);
          }
          REQUIRE(DummyCell::nLive == 2);
        }
        REQUIRE(DummyCell::nLive == 2);
      }
      SECTION("to owning") {
        {
          CaloCellGeometryMayOwnPtr ptr(p);
          auto dummy = std::make_unique<DummyCell>();
          auto dummyAddress = dummy.get();
          REQUIRE(DummyCell::nLive == 2);
          {
            CaloCellGeometryMayOwnPtr from(std::move(dummy));
            {
              ptr = std::move(from);
              CHECK(from.get() == nullptr);
              CHECK(ptr.get() == dummyAddress);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
              CHECK(ptr.operator->() == dummyAddress);
              CHECK(&(*ptr) == dummyAddress);
            }
          }
          REQUIRE(DummyCell::nLive == 2);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
    }
    SECTION("from owning") {
      SECTION("to non-owning") {
        auto oldDummy = std::make_unique<DummyCell>();
        REQUIRE(DummyCell::nLive == 1);
        DummyCell dummy;
        REQUIRE(DummyCell::nLive == 2);
        {
          CaloCellGeometryMayOwnPtr ptr(std::move(oldDummy));
          CaloCellGeometryPtr p(&dummy);
          {
            CaloCellGeometryMayOwnPtr from(p);
            {
              REQUIRE(DummyCell::nLive == 2);
              ptr = std::move(from);
              REQUIRE(DummyCell::nLive == 1);
              CHECK(from.get() == nullptr);
              CHECK(ptr.get() == &dummy);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
              CHECK(ptr.operator->() == &dummy);
              CHECK(&(*ptr) == &dummy);
            }
          }
          REQUIRE(DummyCell::nLive == 1);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
      SECTION("to owning") {
        REQUIRE(DummyCell::nLive == 0);
        {
          auto oldDummy = std::make_unique<DummyCell>();
          REQUIRE(DummyCell::nLive == 1);
          CaloCellGeometryMayOwnPtr ptr(std::move(oldDummy));
          auto dummy = std::make_unique<DummyCell>();
          auto dummyAddress = dummy.get();
          REQUIRE(DummyCell::nLive == 2);
          {
            CaloCellGeometryMayOwnPtr from(std::move(dummy));
            {
              REQUIRE(DummyCell::nLive == 2);
              ptr = std::move(from);
              REQUIRE(DummyCell::nLive == 1);
              CHECK(from.get() == nullptr);
              CHECK(ptr.get() == dummyAddress);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
              CHECK(ptr.operator->() == dummyAddress);
              CHECK(&(*ptr) == dummyAddress);
            }
          }
          REQUIRE(DummyCell::nLive == 1);
        }
        REQUIRE(DummyCell::nLive == 0);
      }
    }
  }

  SECTION("copy assignment") {
    SECTION("from non-owning") {
      DummyCell oldDummy;
      CaloCellGeometryPtr p(&oldDummy);
      REQUIRE(DummyCell::nLive == 1);
      SECTION("to non-owning") {
        DummyCell dummy;
        REQUIRE(DummyCell::nLive == 2);
        {
          CaloCellGeometryMayOwnPtr ptr(p);
          CaloCellGeometryPtr p(&dummy);
          {
            CaloCellGeometryMayOwnPtr from(p);
            {
              ptr = from;
              CHECK(from.get() == &dummy);
              CHECK(ptr.get() == &dummy);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
              CHECK(ptr.operator->() == &dummy);
              CHECK(&(*ptr) == &dummy);
            }
            REQUIRE(DummyCell::nLive == 2);
          }
          REQUIRE(DummyCell::nLive == 2);
        }
        REQUIRE(DummyCell::nLive == 2);
      }
      SECTION("to owning") {
        {
          CaloCellGeometryMayOwnPtr ptr(p);
          auto dummy = std::make_unique<DummyCell>();
          auto dummyAddress = dummy.get();
          REQUIRE(DummyCell::nLive == 2);
          {
            CaloCellGeometryMayOwnPtr from(std::move(dummy));
            {
              ptr = from;
              CHECK(from.get() == dummyAddress);
              CHECK(ptr.get() == dummyAddress);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
              CHECK(ptr.operator->() == dummyAddress);
              CHECK(&(*ptr) == dummyAddress);
            }
          }
          REQUIRE(DummyCell::nLive == 2);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
    }
    SECTION("from owning") {
      SECTION("to non-owning") {
        auto oldDummy = std::make_unique<DummyCell>();
        REQUIRE(DummyCell::nLive == 1);
        DummyCell dummy;
        REQUIRE(DummyCell::nLive == 2);
        {
          CaloCellGeometryMayOwnPtr ptr(std::move(oldDummy));
          CaloCellGeometryPtr p(&dummy);
          {
            CaloCellGeometryMayOwnPtr from(p);
            {
              REQUIRE(DummyCell::nLive == 2);
              ptr = from;
              REQUIRE(DummyCell::nLive == 1);
              CHECK(from.get() == &dummy);
              CHECK(ptr.get() == &dummy);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == &dummy);
              CHECK(ptr.operator->() == &dummy);
              CHECK(&(*ptr) == &dummy);
            }
          }
          REQUIRE(DummyCell::nLive == 1);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
      SECTION("to owning") {
        REQUIRE(DummyCell::nLive == 0);
        {
          auto oldDummy = std::make_unique<DummyCell>();
          REQUIRE(DummyCell::nLive == 1);
          CaloCellGeometryMayOwnPtr ptr(std::move(oldDummy));
          auto dummy = std::make_unique<DummyCell>();
          auto dummyAddress = dummy.get();
          REQUIRE(DummyCell::nLive == 2);
          {
            CaloCellGeometryMayOwnPtr from(std::move(dummy));
            {
              REQUIRE(DummyCell::nLive == 2);
              ptr = from;
              REQUIRE(DummyCell::nLive == 1);
              CHECK(from.get() == dummyAddress);
              CHECK(ptr.get() == dummyAddress);
              CHECK(static_cast<CaloCellGeometry const*>(ptr) == dummyAddress);
              CHECK(ptr.operator->() == dummyAddress);
              CHECK(&(*ptr) == dummyAddress);
            }
          }
          REQUIRE(DummyCell::nLive == 1);
        }
        REQUIRE(DummyCell::nLive == 0);
      }
    }
  }
  SECTION("reference counting") {
    auto dummy = std::make_unique<DummyCell>();
    auto dummyAddress = dummy.get();
    REQUIRE(DummyCell::nLive == 1);
    {
      CaloCellGeometryMayOwnPtr ptr1;
      {
        CaloCellGeometryMayOwnPtr ptr2(std::move(dummy));
        REQUIRE(DummyCell::nLive == 1);
        ptr1 = ptr2;
        REQUIRE(DummyCell::nLive == 1);
        CHECK(ptr1.get() == dummyAddress);
        {
          CaloCellGeometryMayOwnPtr ptr3(ptr1);
          CHECK(ptr3.get() == dummyAddress);
        }
        REQUIRE(DummyCell::nLive == 1);
      }
      REQUIRE(DummyCell::nLive == 1);
    }
    REQUIRE(DummyCell::nLive == 0);
  }
}
