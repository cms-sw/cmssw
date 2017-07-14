#ifndef DDDividedPolyhedra_H
#define DDDividedPolyhedra_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationPolyhedra*
//---------------------------------------------------------------------

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

//---------------------------------------------------------------------
// Class DDDividedPolyhedraRho
//---------------------------------------------------------------------

class DDDividedPolyhedraRho final : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedPolyhedraRho( const DDDivision& div, DDCompactView* cpv );
  
  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

//---------------------------------------------------------------------
// Class DDDividedPolyhedraPhi
//---------------------------------------------------------------------

class DDDividedPolyhedraPhi final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedPolyhedraPhi( const DDDivision& div, DDCompactView* cpv );

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

//---------------------------------------------------------------------
// Class DDDividedPolyhedraZ
//---------------------------------------------------------------------

class DDDividedPolyhedraZ final : public DDDividedGeometryObject
{ 
 public: 

  DDDividedPolyhedraZ( const DDDivision& div, DDCompactView* cpv );

  void checkParametersValidity() override;
  double getMaxParameter() const override;
  DDTranslation makeDDTranslation( const int copyNo ) const override;
  DDRotation makeDDRotation(const int copyNo ) const override;
  DDLogicalPart makeDDLogicalPart( const int copyNo ) const override;
};

#endif
