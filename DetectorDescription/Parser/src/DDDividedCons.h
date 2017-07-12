#ifndef DD_DividedCons_H
#define DD_DividedCons_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationCons*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedConsRho final : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsRho( const DDDivision& div, DDCompactView* cpv );

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

class DDDividedConsPhi final : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsPhi( const DDDivision& div, DDCompactView* cpv );

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

class DDDividedConsZ final : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsZ( const DDDivision& div, DDCompactView* cpv) ;

  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

#endif
