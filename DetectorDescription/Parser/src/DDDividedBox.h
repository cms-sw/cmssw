#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_BOX_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_BOX_H

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedBoxX final : public DDDividedGeometryObject {
public:
  DDDividedBoxX(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedBoxY final : public DDDividedGeometryObject {
public:
  DDDividedBoxY(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

class DDDividedBoxZ final : public DDDividedGeometryObject {
public:
  DDDividedBoxZ(const DDDivision& div, DDCompactView* cpv);

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation(int copyNo) const override;
  DDRotation makeDDRotation(int copyNo) const override;
  DDLogicalPart makeDDLogicalPart(int copyNo) const override;
};

#endif
