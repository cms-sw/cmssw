#ifndef DD_DividedCons_H
#define DD_DividedCons_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationCons*
// ********************************************************************

#include "DetectorDescription/DDParser/interface/DDDividedGeometryObject.h"
#include "DetectorDescription/DDBase/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

class DDDividedConsRho : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsRho( const DDDivision & div );

  virtual ~DDDividedConsRho();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};

class DDDividedConsPhi : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsPhi( const DDDivision & div );

  virtual ~DDDividedConsPhi();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};

class DDDividedConsZ : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedConsZ( const DDDivision & div) ;

  virtual ~DDDividedConsZ();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};
#endif
