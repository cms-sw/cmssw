#include "FWCore/Utilities/interface/stemFromPath.h"

#include <cassert>

int main() {
  assert(edm::stemFromPath("foo.root") == "foo");
  assert(edm::stemFromPath("/foo.root") == "foo");
  assert(edm::stemFromPath("/bar/foo.root") == "foo");
  assert(edm::stemFromPath("/bar///....//...///foo.root") == "foo");
  assert(edm::stemFromPath("/bar/foo.xyzzy") == "foo");
  assert(edm::stemFromPath("/bar/xyzzy.foo.root") == "xyzzy");
  assert(edm::stemFromPath("file:foo.root") == "foo");
  assert(edm::stemFromPath("file:/path/to/bar.txt") == "bar");
  assert(edm::stemFromPath("root://server.somewhere:port/whatever?param=path/to/bar.txt") == "bar");
  return 0;
}
