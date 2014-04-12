#ifndef DDDividedTrd_H
#define DDDividedTrd_H//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationTrd*
// ********************************************************************

#include "DDDividedGeometryObject.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

class DDLogicalPart;
class DDRotation;

class DDDividedTrdX : public DDDividedGeometryObject
{ 
  public:  // with description

    DDDividedTrdX( const DDDivision& div, DDCompactView* cpv );

    virtual ~DDDividedTrdX();

    virtual void checkParametersValidity();

    virtual double getMaxParameter() const;

    virtual DDTranslation makeDDTranslation( const int copyNo) const;
    
    virtual DDRotation makeDDRotation(const int copyNo) const;
    
    virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};


class DDDividedTrdY : public DDDividedGeometryObject
{ 
  public:  // with description

    DDDividedTrdY( const DDDivision& div, DDCompactView* cpv );

    virtual ~DDDividedTrdY();

    virtual void checkParametersValidity();

    virtual double getMaxParameter() const;

    virtual DDTranslation makeDDTranslation( const int copyNo) const;
    
    virtual DDRotation makeDDRotation(const int copyNo) const;
    
    virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;
};


class DDDividedTrdZ : public DDDividedGeometryObject
{ 
  public:  // with description

    DDDividedTrdZ( const DDDivision& div, DDCompactView* cpv );

    virtual ~DDDividedTrdZ();

    virtual void checkParametersValidity();

    virtual double getMaxParameter() const;

    virtual DDTranslation makeDDTranslation( const int copyNo) const;
    
    virtual DDRotation makeDDRotation(const int copyNo) const;
    
    virtual DDLogicalPart makeDDLogicalPart( const int copyNo) const;

};

#endif
