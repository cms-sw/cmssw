#include <iostream>

#include "HeterogeneousCore/AlpakaTest/interface/HostOnlyType.h"

namespace alpakatest {

  void HostOnlyType::print() { std::cout << "The HostOnlyType value is " << value_ << '\n'; }

}  // namespace alpakatest
