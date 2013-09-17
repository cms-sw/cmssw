#include "DetectorDescription/Core/interface/DDAxes.h"
#include <string>
#include <cassert>
#include <iostream>

int
main(int argc, char **argv)
{
  assert(DDAxesNames::index("x") == x);
  assert(DDAxesNames::index("y") == y);
  assert(DDAxesNames::index("z") == z);
  assert(DDAxesNames::index("rho") == rho);
  assert(DDAxesNames::index("radial3D") == radial3D);
  assert(DDAxesNames::index("phi") == phi);
  assert(DDAxesNames::index("undefined") == undefined);
  // If the name does not exists result is always 0.
  //  assert(DDAxesNames::index("foo") == 0);

  assert(DDAxesNames::name(x) == std::string("x"));
  assert(DDAxesNames::name(y) == std::string("y"));
  assert(DDAxesNames::name(z) == std::string("z"));
  assert(DDAxesNames::name(rho) == std::string("rho"));
  assert(DDAxesNames::name(radial3D) == std::string("radial3D"));
  assert(DDAxesNames::name(phi) == std::string("phi"));
  assert(DDAxesNames::name(undefined) == std::string("undefined"));
}
