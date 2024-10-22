#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_TUBS_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_TUBS_H

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedTubsRho final : public DDDividedGeometryObject {
public:
  DDDividedTubsRho(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedTubsPhi final : public DDDividedGeometryObject {
public:
  DDDividedTubsPhi(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedTubsZ final : public DDDividedGeometryObject {
public:
  DDDividedTubsZ(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

#endif
