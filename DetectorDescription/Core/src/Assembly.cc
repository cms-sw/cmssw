#include "DetectorDescription/Core/interface/Assembly.h"

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Solid.h"

DDI::Assembly::Assembly() : Solid(DDSolidShape::ddassembly) {}

void DDI::Assembly::stream(std::ostream& os) const {}
