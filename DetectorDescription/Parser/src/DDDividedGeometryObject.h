#ifndef DD_DividedGeometryObject_H
#define DD_DividedGeometryObject_H
//
// ********************************************************************
// 25.04.04 - M.Case ported algorithm from G4VDivisionParameterisation.hh. to 
//            DDD version
//---------------------------------------------------------------------

#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDDivision.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

enum DivisionType { DivNDIVandWIDTH, DivNDIV, DivWIDTH };

class DDLogicalPart;
class DDRotation;
class DDSolid;

class DDDividedGeometryObject
{ 
public:
  
  DDDividedGeometryObject( const DDDivision& div, DDCompactView* cpv );
  
  virtual ~DDDividedGeometryObject( void );
  
  virtual DDTranslation makeDDTranslation( const int copyNo ) const;
  virtual DDRotation    makeDDRotation   ( const int copyNo ) const ;
  virtual DDLogicalPart makeDDLogicalPart( const int copyNo ) const;

  virtual const std::string& getType( void ) const;

  virtual void setType( const std::string& type );

  int volumeFirstCopyNo( void ) const;

  virtual void execute( void );

  static const double tolerance( void );
  
protected:
  
  DDRotationMatrix* changeRotMatrix( double rotZ = 0. ) const;
  int calculateNDiv( double motherDim, double width,
		     double offset ) const;
  double calculateWidth( double motherDim, int nDiv,
			 double offset ) const;

  virtual void checkParametersValidity( void );

  void checkOffset( double maxPar );
  void checkNDivAndWidth( double maxPar );

  virtual double getMaxParameter( void ) const;

protected:
  DDDivision div_;
  std::string ftype_;
  int compNDiv_;
  double compWidth_;
  DivisionType divisionType_;
  int theVoluFirstCopyNo_;
  DDCompactView* cpv_;
};

#endif
