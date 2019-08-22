#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_POLYCONE_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_POLYCONE_H

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedPolyconeRho final : public DDDividedGeometryObject {
public:
  DDDividedPolyconeRho(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedPolyconePhi final : public DDDividedGeometryObject {
public:
  DDDividedPolyconePhi(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedPolyconeZ final : public DDDividedGeometryObject {
public:
  DDDividedPolyconeZ(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

#endif
