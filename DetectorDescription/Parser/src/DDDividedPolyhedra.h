#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_POLYHEDRA_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_POLYHEDRA_H

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedPolyhedraRho final : public DDDividedGeometryObject {
public:
  DDDividedPolyhedraRho(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedPolyhedraPhi final : public DDDividedGeometryObject {
public:
  DDDividedPolyhedraPhi(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedPolyhedraZ final : public DDDividedGeometryObject {
public:
  DDDividedPolyhedraZ(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

#endif
