#include "DetectorDescription/Parser/src/DDDividedGeometryObject.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/GenVector/RotationZ.h"

#include <iostream>
#include <utility>

DDDividedGeometryObject::DDDividedGeometryObject( const DDDivision& div, DDCompactView* cpv ) 
  : div_( div ),
    ftype_(),
    compNDiv_( div.nReplicas()),
    compWidth_( div.width()),
    divisionType_( DivNDIVandWIDTH ),
    theVoluFirstCopyNo_( 1 ),
    cpv_( cpv )
{
  if( div_.nReplicas() == 0 || div_.width() < tolerance())
  {
    if( div_.width() < tolerance())
      divisionType_ = DivNDIV;
    else 
      divisionType_ = DivWIDTH;
  } 
}

std::unique_ptr<DDRotationMatrix>
DDDividedGeometryObject::changeRotMatrix( double rotZ ) const
{
  return std::make_unique<DDRotationMatrix>(ROOT::Math::RotationZ(rotZ));
}

int
DDDividedGeometryObject::calculateNDiv( double motherDim, double width, double offset ) const
{
  return int( ( motherDim - offset ) / width );
}

double
DDDividedGeometryObject::calculateWidth( double motherDim, int nDiv, double offset ) const
{ 
  return ( motherDim - offset ) / nDiv;
}

void
DDDividedGeometryObject::checkParametersValidity( void )
{
  double maxPar = getMaxParameter();
  checkOffset( maxPar );
  checkNDivAndWidth( maxPar );
  if (!div_.parent().isDefined().second) {
    std::string s = "DDDividedGeometryObject::checkParametersValidity() :";
    s+= "\n ERROR - the LogicalPart of the parent must be ";
    s+= "\n         defined before a division can occur.";
    s+= "\n         Parent= " + div_.parent().toString();
    throw cms::Exception("DDException") << s;
  }
}

void
DDDividedGeometryObject::checkOffset( double maxPar )
{
  if( div_.offset() >= maxPar )
  {
    std::string s = "DDDividedGeometryObject::checkOffset() IllegalConstruct";
    s += "\nERROR - DDDividedGeometryObject::checkOffset()";
    s += "\n        failed.";
    s += "  Too big an offset.";
    throw cms::Exception("DDException") << s;
  }
}

void
DDDividedGeometryObject::checkNDivAndWidth( double maxPar )
{
  if( (divisionType_ == DivNDIVandWIDTH)
      && (div_.offset() + compWidth_*compNDiv_ - maxPar > tolerance() ) )
  {
    std::string s = "ERROR - DDDividedGeometryObject::checkNDivAndWidth()";
    s+= "\n        Division of LogicalPart " + div_.parent().name().name();
    s+= " has too big an offset.";

    std::cout << compWidth_ << std::endl;
    throw cms::Exception("DDException") << s;
  }
}

const double
DDDividedGeometryObject::tolerance( void )
{
  // this can come from some global tolerance if you want.
  static const double tol = 1.0/1000.00;
  return tol;
}

void
DDDividedGeometryObject::setType( const std::string& s) 
{
  ftype_ = s;
}

const std::string&
DDDividedGeometryObject::getType( void ) const
{
  return ftype_;
}

void
DDDividedGeometryObject::execute( void )
{
  for( int i = theVoluFirstCopyNo_; i < compNDiv_+theVoluFirstCopyNo_; ++i )
  {
    cpv_->position( makeDDLogicalPart( i ),
		    div_.parent(),
		    i,
		    makeDDTranslation( i ),
		    makeDDRotation( i ),
		    &div_ );
  }
}

double
DDDividedGeometryObject::getMaxParameter( void ) const
{
  return 0.0;
}

DDRotation
DDDividedGeometryObject::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDTranslation
DDDividedGeometryObject::makeDDTranslation( const int copyNo ) const
{
  return DDTranslation();
}

DDLogicalPart
DDDividedGeometryObject::makeDDLogicalPart( const int copyNo ) const
{
  // just return the parent... this is USELESS
  return div_.parent();
}
