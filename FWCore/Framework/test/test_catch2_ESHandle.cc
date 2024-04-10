#include "catch.hpp"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace {
  class TestExceptionFactory : public edm::ESHandleExceptionFactory {
  public:
    std::exception_ptr make() const override { return std::make_exception_ptr(edm::Exception(edm::errors::OtherCMS)); }
  };
}  // namespace

TEST_CASE("test edm::ESHandle", "[ESHandle]") {
  SECTION("Default constructor") {
    edm::ESHandle<int> handle;
    REQUIRE(not handle.isValid());
    REQUIRE(not handle.failedToGet());
    REQUIRE_THROWS_AS(handle.description(), edm::Exception);
    REQUIRE(handle.product() == nullptr);
  }

  SECTION("Valid construction") {
    int const value = 42;

    SECTION("without ComponentDescription") {
      edm::ESHandle<int> handle(&value);
      REQUIRE(not handle.isValid());
      REQUIRE(not handle.failedToGet());
      REQUIRE_THROWS_AS(handle.description(), edm::Exception);
      REQUIRE(handle.product() != nullptr);
      REQUIRE(*handle == value);
    }

    SECTION("Valid construction, with ComponentDescription") {
      edm::eventsetup::ComponentDescription const desc;
      edm::ESHandle<int> handle(&value, &desc);
      REQUIRE(handle.isValid());
      REQUIRE(not handle.failedToGet());
      REQUIRE(handle.description() == &desc);
      REQUIRE(handle.product() != nullptr);
      REQUIRE(*handle == value);
    }
  }

  SECTION("Construction for a 'failure'") {
    SECTION("From temporary factory object") {
      edm::ESHandle<int> handle(std::make_shared<TestExceptionFactory>());
      REQUIRE(not handle.isValid());
      REQUIRE(handle.failedToGet());
      REQUIRE_THROWS_AS(handle.description(), edm::Exception);
      REQUIRE_THROWS_AS(handle.product(), edm::Exception);
    }

    SECTION("From another factory object") {
      auto const factory = std::make_shared<TestExceptionFactory>();
      edm::ESHandle<int> handle(factory);
      REQUIRE(not handle.isValid());
      REQUIRE(handle.failedToGet());
      REQUIRE_THROWS_AS(handle.description(), edm::Exception);
      REQUIRE_THROWS_AS(handle.product(), edm::Exception);
      REQUIRE(handle.whyFailedFactory().get() == factory.get());
    }

    SECTION("From another ESHandle") {
      auto const factory = std::make_shared<TestExceptionFactory>();
      edm::ESHandle<int> handleA(factory);
      edm::ESHandle<int> handle(handleA.whyFailedFactory());
      REQUIRE(not handle.isValid());
      REQUIRE(handle.failedToGet());
      REQUIRE_THROWS_AS(handle.description(), edm::Exception);
      REQUIRE_THROWS_AS(handle.product(), edm::Exception);
      REQUIRE(handle.whyFailedFactory().get() == factory.get());
    }
  }

  SECTION("Copying") {
    int const valueA = 42;
    edm::eventsetup::ComponentDescription const descA;

    SECTION("From valid ESHandle") {
      edm::ESHandle<int> const handleA(&valueA, &descA);

      SECTION("Constructor") {
        edm::ESHandle<int> handleB(handleA);
        REQUIRE(handleA.isValid());
        REQUIRE(*handleA == valueA);
        REQUIRE(handleB.isValid());
        REQUIRE(*handleB == valueA);
      }

      SECTION("Assignment") {
        edm::ESHandle<int> handleB;
        REQUIRE(not handleB.isValid());

        handleB = handleA;
        REQUIRE(handleA.isValid());
        REQUIRE(*handleA == valueA);
        REQUIRE(handleB.isValid());
        REQUIRE(*handleB == valueA);
      }
    }

    SECTION("From invalid ESHandle") {
      edm::ESHandle<int> const handleA(std::make_shared<TestExceptionFactory>());

      SECTION("Constructor") {
        edm::ESHandle<int> handleB(handleA);
        REQUIRE(not handleA.isValid());
        REQUIRE(handleA.failedToGet());
        REQUIRE_THROWS_AS(handleA.description(), edm::Exception);
        REQUIRE_THROWS_AS(handleA.product(), edm::Exception);

        REQUIRE(not handleB.isValid());
        REQUIRE(handleB.failedToGet());
        REQUIRE_THROWS_AS(handleB.description(), edm::Exception);
        REQUIRE_THROWS_AS(handleB.product(), edm::Exception);
      }

      SECTION("Assignment") {
        edm::ESHandle<int> handleB(&valueA, &descA);
        REQUIRE(handleB.isValid());

        handleB = handleA;
        REQUIRE(not handleA.isValid());
        REQUIRE(handleA.failedToGet());
        REQUIRE_THROWS_AS(handleA.description(), edm::Exception);
        REQUIRE_THROWS_AS(handleA.product(), edm::Exception);

        REQUIRE(not handleB.isValid());
        REQUIRE(handleB.failedToGet());
        REQUIRE_THROWS_AS(handleB.description(), edm::Exception);
        REQUIRE_THROWS_AS(handleB.product(), edm::Exception);
      }
    }
  }

  SECTION("Moving") {
    int const valueA = 42;
    edm::eventsetup::ComponentDescription const descA;

    SECTION("From valid ESHandle") {
      edm::ESHandle<int> handleA(&valueA, &descA);

      SECTION("Constructor") {
        edm::ESHandle<int> handleB(std::move(handleA));
        REQUIRE(handleB.isValid());
        REQUIRE(*handleB == valueA);
      }

      SECTION("Assignment") {
        edm::ESHandle<int> handleB;
        REQUIRE(not handleB.isValid());

        handleB = std::move(handleA);
        REQUIRE(handleB.isValid());
        REQUIRE(*handleB == valueA);
      }
    }

    SECTION("From invalid ESHandle") {
      edm::ESHandle<int> handleA(std::make_shared<TestExceptionFactory>());

      SECTION("Constructor") {
        edm::ESHandle<int> handleB(std::move(handleA));
        // this is pretty much the only feature that we can test on
        // the moved-from ESHandle that is guaranteed to change (to
        // test that move actually happens instead of copy)
        REQUIRE(not handleA.failedToGet());

        REQUIRE(not handleB.isValid());
        REQUIRE(handleB.failedToGet());
        REQUIRE_THROWS_AS(handleB.description(), edm::Exception);
        REQUIRE_THROWS_AS(handleB.product(), edm::Exception);
      }

      SECTION("Assignment") {
        edm::ESHandle<int> handleB(&valueA, &descA);
        REQUIRE(handleB.isValid());

        handleB = std::move(handleA);
        // this is pretty much the only feature that we can test on
        // the moved-from ESHandle that is guaranteed to change (to
        // test that move actually happens instead of copy)
        REQUIRE(not handleA.failedToGet());

        REQUIRE(not handleB.isValid());
        REQUIRE(handleB.failedToGet());
        REQUIRE_THROWS_AS(handleB.description(), edm::Exception);
        REQUIRE_THROWS_AS(handleB.product(), edm::Exception);
      }
    }
  }

  SECTION("Swap") {
    int const valueA = 42;
    edm::eventsetup::ComponentDescription const descA;
    edm::ESHandle<int> handleA(&valueA, &descA);

    SECTION("With value") {
      int const valueB = 3;
      edm::ESHandle<int> handleB(&valueB);

      std::swap(handleA, handleB);
      REQUIRE(not handleA.isValid());
      REQUIRE(handleB.isValid());
      REQUIRE(*handleB == valueA);
    }

    SECTION("With failure factory") {
      auto factory = std::make_shared<TestExceptionFactory>();
      edm::ESHandle<int> handleB(factory);

      std::swap(handleA, handleB);
      REQUIRE(not handleA.isValid());
      REQUIRE(handleA.failedToGet());
      REQUIRE(handleA.whyFailedFactory().get() == factory.get());
      REQUIRE(handleB.isValid());
      REQUIRE(*handleB == valueA);
    }
  }
}
