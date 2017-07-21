#ifndef DD_DividedTubs_H
#define DD_DividedTubs_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize  G4ParameterisationTubs*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedTubsRho final : public DDDividedGeometryObject
{ 
 public:

  DDDividedTubsRho( const DDDivision& div, DDCompactView* cpv );

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation( const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart(const int copyNo ) const override;
};

class DDDividedTubsPhi final : public DDDividedGeometryObject
{ 
 public:

  DDDividedTubsPhi( const DDDivision& div, DDCompactView* cpv );
  
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation( const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart(const int copyNo ) const override;
};

class DDDividedTubsZ final : public DDDividedGeometryObject
{ 
 public:

  DDDividedTubsZ( const DDDivision& div, DDCompactView* cpv );

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation( const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart(const int copyNo ) const override;
};

#endif
