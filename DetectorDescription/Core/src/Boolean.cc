#include "DetectorDescription/Core/interface/Boolean.h"
#include "DetectorDescription/Core/interface/Solid.h"

DDI::BooleanSolid::BooleanSolid(
    const DDSolid& A, const DDSolid& B, const DDTranslation& t, const DDRotation& r, DDSolidShape s)
    : Solid(s), a_(A), b_(B), t_(t), r_(r) {}

DDI::Union::Union(const DDSolid& A, const DDSolid& B, const DDTranslation& t, const DDRotation& r)
    : DDI::BooleanSolid(A, B, t, r, DDSolidShape::ddunion) {}

DDI::Intersection::Intersection(const DDSolid& A, const DDSolid& B, const DDTranslation& t, const DDRotation& r)
    : DDI::BooleanSolid(A, B, t, r, DDSolidShape::ddintersection) {}

DDI::Subtraction::Subtraction(const DDSolid& A, const DDSolid& B, const DDTranslation& t, const DDRotation& r)
    : DDI::BooleanSolid(A, B, t, r, DDSolidShape::ddsubtraction) {}
