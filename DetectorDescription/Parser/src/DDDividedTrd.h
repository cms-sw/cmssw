#ifndef DDDividedTrd_H
#define DDDividedTrd_H//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationTrd*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedTrdX final : public DDDividedGeometryObject
{ 
  public:

    DDDividedTrdX( const DDDivision& div, DDCompactView* cpv );

    void checkParametersValidity() override;
    double getMaxParameter() const override;
    DDTranslation makeDDTranslation( const int copyNo ) const override;
    DDRotation makeDDRotation(const int copyNo ) const override;
    DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

class DDDividedTrdY final : public DDDividedGeometryObject
{ 
  public:

    DDDividedTrdY( const DDDivision& div, DDCompactView* cpv );

    void checkParametersValidity() override;
    double getMaxParameter() const override;
    DDTranslation makeDDTranslation( const int copyNo ) const override;
    DDRotation makeDDRotation(const int copyNo ) const override;
    DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

class DDDividedTrdZ final : public DDDividedGeometryObject
{
  public:

    DDDividedTrdZ( const DDDivision& div, DDCompactView* cpv );

    void checkParametersValidity() override;
    double getMaxParameter() const override;
    DDTranslation makeDDTranslation( const int copyNo ) const override;
    DDRotation makeDDRotation(const int copyNo ) const override;
    DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

#endif
