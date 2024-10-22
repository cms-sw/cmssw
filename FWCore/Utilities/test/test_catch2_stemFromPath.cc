#include "FWCore/Utilities/interface/stemFromPath.h"

#include "catch.hpp"

TEST_CASE("Test stemFromPath", "[sources]") {
  CHECK(edm::stemFromPath("foo.root") == "foo");
  CHECK(edm::stemFromPath("/foo.root") == "foo");
  CHECK(edm::stemFromPath("/bar/foo.root") == "foo");
  CHECK(edm::stemFromPath("/bar///....//...///foo.root") == "foo");
  CHECK(edm::stemFromPath("/bar/foo.xyzzy") == "foo");
  CHECK(edm::stemFromPath("/bar/xyzzy.foo.root") == "xyzzy");
  CHECK(edm::stemFromPath("file:foo.root") == "foo");
  CHECK(edm::stemFromPath("file:/path/to/bar.txt") == "bar");
  CHECK(edm::stemFromPath("root://server.somewhere:port/whatever?param=path/to/bar.txt") == "bar");
}
