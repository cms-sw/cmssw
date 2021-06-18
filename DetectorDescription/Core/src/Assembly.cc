#include "DetectorDescription/Core/src/Assembly.h"

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

DDI::Assembly::Assembly() : Solid(DDSolidShape::ddassembly) {}

void DDI::Assembly::stream(std::ostream& os) const {}
