#ifndef DD_DividedCons_H
#define DD_DividedCons_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationCons*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

class DDDividedConsRho : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsRho( const DDDivision& div, DDCompactView* cpv );

  virtual ~DDDividedConsRho();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};

class DDDividedConsPhi : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsPhi( const DDDivision& div, DDCompactView* cpv );

  virtual ~DDDividedConsPhi();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};

class DDDividedConsZ : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsZ( const DDDivision& div, DDCompactView* cpv) ;

  virtual ~DDDividedConsZ();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};
#endif
