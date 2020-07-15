/*
 *  CMSSW
 *
 */

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"

class CopyNoMove {
public:
  CopyNoMove() {}
  CopyNoMove(CopyNoMove const&) { /* std::cout << "copied\n"; */
  }
  CopyNoMove& operator=(CopyNoMove const&) { /*std::cout << "assigned\n";*/
    return *this;
  }

private:
};

class MoveNoCopy {
public:
  MoveNoCopy() {}
  MoveNoCopy(MoveNoCopy const&) = delete;
  MoveNoCopy& operator=(MoveNoCopy const&) = delete;
  MoveNoCopy(MoveNoCopy&&) { /* std::cout << "moved\n";*/
  }
  MoveNoCopy& operator=(MoveNoCopy&&) { /* std::cout << "moved\n";*/
    return *this;
  }

private:
};

void work() {
  auto thing = std::make_unique<CopyNoMove>();
  edm::Wrapper<CopyNoMove> wrap(std::move(thing));

  auto thing2 = std::make_unique<MoveNoCopy>();
  edm::Wrapper<MoveNoCopy> wrap2(std::move(thing2));

  auto thing3 = std::make_unique<std::vector<double>>(10, 2.2);
  assert(thing3->size() == 10);

  edm::Wrapper<std::vector<double>> wrap3(std::move(thing3));
  assert(wrap3->size() == 10);
  assert(thing3.get() == nullptr);
}

int main() {
  int rc = 0;
  try {
    work();
  } catch (...) {
    rc = 1;
    std::cerr << "Failure: unidentified exception caught\n";
  }
  return rc;
}
