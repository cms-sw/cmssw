#ifndef DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_GEOMETRY_OBJECT_H
#define DETECTOR_DESCRIPTION_PARSER_DD_DIVIDED_GEOMETRY_OBJECT_H

#include <string>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;

enum DivisionType { DivNDIVandWIDTH, DivNDIV, DivWIDTH };

class DDLogicalPart;
class DDRotation;
class DDSolid;

class DDDividedGeometryObject
{ 
public:
  
  DDDividedGeometryObject( const DDDivision& div, DDCompactView* cpv );
  
  virtual ~DDDividedGeometryObject( void ) = default; // inline
  
  virtual DDTranslation makeDDTranslation( int copyNo ) const;
  virtual DDRotation    makeDDRotation   ( int copyNo ) const;
  virtual DDLogicalPart makeDDLogicalPart( int copyNo ) const;

  virtual const std::string& getType( void ) const;

  virtual void setType( const std::string& type );

  int volumeFirstCopyNo( void ) const;

  virtual void execute( void );

  static const double tolerance( void );
  
protected:
  
  std::unique_ptr<DDRotationMatrix> changeRotMatrix( double rotZ = 0. ) const;
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
