#ifndef DD_DividedPolycone_H
#define DD_DividedPolycone_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationPolycone*
//---------------------------------------------------------------------

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

//---------------------------------------------------------------------
// Class DDDividedPolycone
//---------------------------------------------------------------------

class DDDividedPolyconeRho final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedPolyconeRho( const DDDivision& div, DDCompactView* cpv );
  
  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

class DDDividedPolyconePhi final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedPolyconePhi( const DDDivision& div, DDCompactView* cpv );
  
  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

class DDDividedPolyconeZ final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedPolyconeZ( const DDDivision& div, DDCompactView* cpv );
  
  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation( const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

#endif
