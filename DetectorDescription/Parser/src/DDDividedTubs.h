#ifndef DD_DividedTubs_H
#define DD_DividedTubs_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize  G4ParameterisationTubs*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

class DDDividedTubsRho : public DDDividedGeometryObject
{ 
 public:

  DDDividedTubsRho( const DDDivision& div, DDCompactView* cpv);

  virtual ~DDDividedTubsRho();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo ) const;

  virtual DDRotation makeDDRotation( const int copyNo ) const;

  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const;

};

class DDDividedTubsPhi : public DDDividedGeometryObject
{ 
 public:

  DDDividedTubsPhi( const DDDivision& div, DDCompactView* cpv);

  virtual ~DDDividedTubsPhi();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo ) const;

  virtual DDRotation makeDDRotation( const int copyNo ) const;

  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const;

};

class DDDividedTubsZ : public DDDividedGeometryObject
{ 
 public:

  DDDividedTubsZ( const DDDivision& div, DDCompactView* cpv);

  virtual ~DDDividedTubsZ();
  
  virtual double getMaxParameter() const;

  virtual DDTranslation makeDDTranslation( const int copyNo ) const;

  virtual DDRotation makeDDRotation( const int copyNo ) const;

  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const;

};

#endif
