#ifndef DD_DividedBox_H
#define DD_DividedBox_H

//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParametarisationBox*
// ********************************************************************

#include "DetectorDescription/DDParser/interface/DDDividedGeometryObject.h"
#include "DetectorDescription/DDBase/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

class DDDividedBoxX : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedBoxX( const DDDivision & div);
  
  virtual ~DDDividedBoxX();

  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo ) const;

  virtual DDRotation makeDDRotation( const int copyNo ) const;

  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const;
};

class DDDividedBoxY : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedBoxY( const DDDivision & div);
  
  virtual ~DDDividedBoxY();

  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo ) const;

  virtual DDRotation makeDDRotation( const int copyNo ) const;

  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const;
};

class DDDividedBoxZ : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedBoxZ( const DDDivision & div);
  
  virtual ~DDDividedBoxZ();

  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo ) const;

  virtual DDRotation makeDDRotation( const int copyNo ) const;

  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const;
};
#endif
