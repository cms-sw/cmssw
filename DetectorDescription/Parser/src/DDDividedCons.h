#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_CONS_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_CONS_H

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedConsRho final : public DDDividedGeometryObject {
public:
  DDDividedConsRho(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedConsPhi final : public DDDividedGeometryObject {
public:
  DDDividedConsPhi(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedConsZ final : public DDDividedGeometryObject {
public:
  DDDividedConsZ(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

#endif
