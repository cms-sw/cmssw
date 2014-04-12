#include "DetectorDescription/Core/interface/DDTransform.h"

int
main(int argc, char **argv)
{
  assert(DDRotation().toString() == "DdBlNa:DdBlNa0");
  assert(DDRotation().toString() == "DdBlNa:DdBlNa1");
}
