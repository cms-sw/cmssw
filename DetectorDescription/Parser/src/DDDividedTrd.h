#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_TRD_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_TRD_H

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedTrdX final : public DDDividedGeometryObject {
public:
  DDDividedTrdX(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedTrdY final : public DDDividedGeometryObject {
public:
  DDDividedTrdY(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedTrdZ final : public DDDividedGeometryObject {
public:
  DDDividedTrdZ(const DDDivision& div, DDCompactView* cpv);

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

#endif
