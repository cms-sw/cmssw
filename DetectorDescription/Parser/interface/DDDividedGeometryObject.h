#ifndef DD_DividedGeometryObject_H
#define DD_DividedGeometryObject_H
//
// ********************************************************************
// 25.04.04 - M.Case ported algorithm from G4VDivisionParameterisation.hh. to 
//            DDD version
//---------------------------------------------------------------------

#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

enum DivisionType { DivNDIVandWIDTH, DivNDIV, DivWIDTH };

class DDLogicalPart;
class DDRotation;
class DDSolid;
class DDPosPart;


class DDDividedGeometryObject //: public DDDivision 
{ 
 public:
  
  DDDividedGeometryObject( const DDDivision & div );
  
  virtual ~DDDividedGeometryObject();
  
  virtual DDTranslation makeDDTranslation( const int copyNo ) const;
  virtual DDRotation    makeDDRotation   ( const int copyNo ) const ;
  virtual DDLogicalPart makeDDLogicalPart( const int copyNo ) const;

  virtual const string& getType() const;

  virtual void setType(const string& type);

  int volumeFirstCopyNo() const;

  virtual void execute();

  static const double tolerance();
  
 protected:
  
    DDRotationMatrix* changeRotMatrix( double rotZ = 0. ) const;
    int calculateNDiv( double motherDim, double width,
		       double offset ) const;
    double calculateWidth( double motherDim, int nDiv,
			   double offset ) const;

    virtual void checkParametersValidity();

    void checkOffset( double maxPar );
    void checkNDivAndWidth( double maxPar );

    virtual double getMaxParameter() const;

 protected:
    DDDivision div_;
    string ftype_;
    int compNDiv_;
    double compWidth_;
    DivisionType divisionType_;
    int theVoluFirstCopyNo_;
};

#endif
