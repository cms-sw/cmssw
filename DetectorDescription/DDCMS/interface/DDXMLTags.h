#ifndef DETECTOR_DESCRIPTION_DD_XML_TAGS_H
#define DETECTOR_DESCRIPTION_DD_XML_TAGS_H

#include "XML/XMLElements.h"
#ifndef UNICODE
#define UNICODE(x) extern const ::dd4hep::xml::Tag_t Unicode_##x
#endif

namespace cms {

  UNICODE(DDCMS);

  UNICODE(DDDefinition);

  UNICODE(ConstantsSection);
  UNICODE(Constant);

  UNICODE(MaterialSection);
  UNICODE(ElementaryMaterial);
  UNICODE(CompositeMaterial);
  UNICODE(atomicWeight);
  UNICODE(density);
  UNICODE(symbol);
  UNICODE(atomicNumber);
  UNICODE(MaterialFraction);

  UNICODE(RotationSection);
  UNICODE(Rotation);
  UNICODE(ReflectionRotation);
  UNICODE(RotationSequence);
  UNICODE(RotationByAxis);
  UNICODE(axis);
  UNICODE(rRotation);
  UNICODE(rReflectionRotation);
  UNICODE(thetaX);
  UNICODE(phiX);
  UNICODE(thetaY);
  UNICODE(phiY);
  UNICODE(thetaZ);
  UNICODE(phiZ);

  UNICODE(TransformationSection);
  UNICODE(Transformation);

  UNICODE(SolidSection);

  UNICODE(PseudoTrap);
  UNICODE(dx1);
  UNICODE(dy1);
  UNICODE(dx2);
  UNICODE(dy2);
  UNICODE(atMinusZ);

  UNICODE(Box);
  UNICODE(dx);
  UNICODE(dy);
  UNICODE(dz);

  UNICODE(Cone);
  UNICODE(rMin1);
  UNICODE(rMax1);
  UNICODE(rMin2);
  UNICODE(rMax2);

  UNICODE(Tubs);
  UNICODE(rMin);
  UNICODE(rMax);
  UNICODE(startPhi);
  UNICODE(deltaPhi);

  UNICODE(Polycone);
  UNICODE(ZSection);
  UNICODE(RZPoint);

  UNICODE(ZXYSection);
  UNICODE(XYPoint);
  UNICODE(scale);

  UNICODE(CutTubs);
  UNICODE(lx);
  UNICODE(ly);
  UNICODE(lz);
  UNICODE(tx);
  UNICODE(ty);
  UNICODE(tz);

  UNICODE(TruncTubs);
  UNICODE(cutAtStart);
  UNICODE(cutAtDelta);
  UNICODE(cutInside);
  UNICODE(zHalf);

  UNICODE(Polyhedra);
  UNICODE(numSide);

  UNICODE(Trapezoid);
  UNICODE(alp1);
  UNICODE(h1);
  UNICODE(bl1);
  UNICODE(tl1);
  UNICODE(alp2);
  UNICODE(h2);
  UNICODE(bl2);
  UNICODE(tl2);

  UNICODE(Sphere);
  UNICODE(startTheta);
  UNICODE(deltaTheta);

  UNICODE(Ellipsoid);
  UNICODE(xSemiAxis);
  UNICODE(ySemiAxis);
  UNICODE(zSemiAxis);
  UNICODE(zBottomCut);
  UNICODE(zTopCut);

  UNICODE(EllipticalTube);
  UNICODE(zHeight);

  UNICODE(Torus);
  UNICODE(torusRadius);
  UNICODE(innerRadius);
  UNICODE(outerRadius);

  UNICODE(SubtractionSolid);
  UNICODE(firstSolid);
  UNICODE(secondSolid);

  UNICODE(LogicalPartSection);
  UNICODE(LogicalPart);
  UNICODE(rSolid);
  UNICODE(rMaterial);

  UNICODE(PosPartSection);
  UNICODE(PosPart);
  UNICODE(copyNumber);
  UNICODE(rParent);
  UNICODE(ChildName);
  UNICODE(rChild);
  UNICODE(Translation);

  UNICODE(SpecParSection);
  UNICODE(SpecPar);
  UNICODE(PartSelector);
  UNICODE(Parameter);
  UNICODE(path);
  UNICODE(value);

  UNICODE(Division);
  UNICODE(parent);
  UNICODE(offset);
  UNICODE(width);
  UNICODE(nReplicas);

  UNICODE(Algorithm);
  UNICODE(String);
  UNICODE(Numeric);
  UNICODE(Vector);
  UNICODE(nEntries);

  UNICODE(debug_constants);
  UNICODE(debug_materials);
  UNICODE(debug_shapes);
  UNICODE(debug_volumes);
  UNICODE(debug_placements);
  UNICODE(debug_namespaces);
  UNICODE(debug_rotations);
  UNICODE(debug_includes);
  UNICODE(debug_algorithms);
  UNICODE(debug_specpars);

  /// DD4hep specific
  UNICODE(open_geometry);
  UNICODE(close_geometry);
  UNICODE(IncludeSection);
  UNICODE(Include);

}  // namespace cms

#undef UNICODE  // Do not miss this one!
#include "XML/XMLTags.h"

#define DD_CMU(a) ::cms::Unicode_##a

#endif
