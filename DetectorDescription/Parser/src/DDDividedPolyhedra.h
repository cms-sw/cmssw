#ifndef DDDividedPolyhedra_H
#define DDDividedPolyhedra_H
//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationPolyhedra*
//---------------------------------------------------------------------

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

//---------------------------------------------------------------------
// Class DDDividedPolyhedraRho
//---------------------------------------------------------------------

class DDDividedPolyhedraRho : public DDDividedGeometryObject
{ 
 public:  
  
  DDDividedPolyhedraRho( const DDDivision& div, DDCompactView* cpv );
  
  virtual ~DDDividedPolyhedraRho();
  
  virtual void checkParametersValidity();
  
  virtual double getMaxParameter() const;
  
  virtual DDTranslation makeDDTranslation( const int copyNo) const;
  
  virtual DDRotation makeDDRotation(const int copyNo) const;
  
  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};

//---------------------------------------------------------------------
// Class DDDividedPolyhedraPhi
//---------------------------------------------------------------------

class DDDividedPolyhedraPhi : public DDDividedGeometryObject
{ 
 public:
  
  DDDividedPolyhedraPhi( const DDDivision& div, DDCompactView* cpv );

  virtual ~DDDividedPolyhedraPhi();

  virtual void checkParametersValidity();
  
  virtual double getMaxParameter() const;
  
  virtual DDTranslation makeDDTranslation( const int copyNo) const;
  
  virtual DDRotation makeDDRotation(const int copyNo) const;
  
  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};

//---------------------------------------------------------------------
// Class DDDividedPolyhedraZ
//---------------------------------------------------------------------

class DDDividedPolyhedraZ : public DDDividedGeometryObject
{ 
 public: 

  DDDividedPolyhedraZ( const DDDivision& div, DDCompactView* cpv );

  virtual ~DDDividedPolyhedraZ();

  virtual void checkParametersValidity();
  
  virtual double getMaxParameter() const;
  
  virtual DDTranslation makeDDTranslation( const int copyNo) const;
  
  virtual DDRotation makeDDRotation(const int copyNo) const;
  
  virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};

#endif
