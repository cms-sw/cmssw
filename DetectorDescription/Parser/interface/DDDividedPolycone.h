#ifndef DD_DividedPolycone_H
#define DD_DividedPolycone_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationPolycone*
//---------------------------------------------------------------------

#include "DetectorDescription/DDParser/interface/DDDividedGeometryObject.h"
#include "DetectorDescription/DDBase/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

//---------------------------------------------------------------------
// Class DDDividedPolycone
//---------------------------------------------------------------------

class DDDividedPolyconeRho : public DDDividedGeometryObject
{ 
 public:  // with description
  
  DDDividedPolyconeRho( const DDDivision & div );
  
  virtual ~DDDividedPolyconeRho();
  
  virtual void checkParametersValidity();
  
  virtual double getMaxParameter() const;
  
  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};

class DDDividedPolyconePhi : public DDDividedGeometryObject
{ 
 public:  // with description
  
  DDDividedPolyconePhi( const DDDivision & div );
  
  virtual ~DDDividedPolyconePhi();
  
  virtual void checkParametersValidity();
  
  virtual double getMaxParameter() const;
  
  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};

class DDDividedPolyconeZ : public DDDividedGeometryObject
{ 
 public:  // with description
  
  DDDividedPolyconeZ( const DDDivision & div );
  
  virtual ~DDDividedPolyconeZ();
  
  virtual void checkParametersValidity();
  
  virtual double getMaxParameter() const;
  
  virtual DDTranslation makeDDTranslation( const int copyNo) const;

  virtual DDRotation makeDDRotation(const int copyNo) const;

  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};

#endif
