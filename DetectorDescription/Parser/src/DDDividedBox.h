#ifndef DD_DividedBox_H
#define DD_DividedBox_H

//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParametarisationBox*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDLogicalPart;
class DDRotation;

class DDDividedBoxX final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedBoxX( const DDDivision& div, DDCompactView* cpv);
  
  virtual double getMaxParameter() const override;
  virtual DDTranslation makeDDTranslation( const int copyNo ) const override;
  virtual DDRotation makeDDRotation( const int copyNo ) const override;
  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const override;
};

class DDDividedBoxY final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedBoxY( const DDDivision& div, DDCompactView* cpv);
  
  virtual double getMaxParameter() const override;
  virtual DDTranslation makeDDTranslation( const int copyNo ) const override;
  virtual DDRotation makeDDRotation( const int copyNo ) const override;
  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const override;
};

class DDDividedBoxZ final : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedBoxZ( const DDDivision& div, DDCompactView* cpv);
  
  virtual double getMaxParameter() const override;
  virtual DDTranslation makeDDTranslation( const int copyNo ) const override;
  virtual DDRotation makeDDRotation( const int copyNo ) const override;
  virtual DDLogicalPart makeDDLogicalPart(const int copyNo) const override;
};
#endif
