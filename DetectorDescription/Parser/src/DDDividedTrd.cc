#include "DetectorDescription/Parser/src/DDDividedTrd.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Parser/src/DDDividedGeometryObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>
#include <ostream>
#include <string>
#include <utility>

class DDCompactView;

using namespace dd::operators;

DDDividedTrdX::DDDividedTrdX( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject(div,cpv)
{
  checkParametersValidity();
  setType( "DivisionTrdX" );
  DDTrap mtrd = (DDTrap)( div_.parent().solid() );

  if ( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( 2 * mtrd.x1(), div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ = calculateWidth( 2*mtrd.x1(), div_.nReplicas(), div_.offset() );
  }
}

double
DDDividedTrdX::getMaxParameter( void ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  return 2 * mtrd.x1();
}

DDTranslation
DDDividedTrdX::makeDDTranslation( const int copyNo ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  double mdx = mtrd.x1();

  //----- translation 
  double posi = -mdx + div_.offset() + (copyNo+0.5)*compWidth_;

  if( div_.axis() == DDAxes::x )
  {
    return DDTranslation(posi, 0.0, 0.0);
  }
  else
  { 
    std::string s = "ERROR - DDDividedTrdX::makeDDTranslation()";
    s += "\n        Axis is along ";
    s += DDAxesNames::name(div_.axis());
    s += " !\n" ;
    s += "DDDividedTrdX::makeDDTranslation()";
    s += " IllegalConstruct: Only axes along x are allowed !";
    throw cms::Exception("DDException") << s;
  }
  
  return DDTranslation();
}

DDRotation
DDDividedTrdX::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDLogicalPart
DDDividedTrdX::makeDDLogicalPart( const int copyNo ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  DDMaterial usemat = div_.parent().material();

  double pDy1 = mtrd.y1(); // Y half length1
  double pDy2 = mtrd.y2(); // Y half length2
  double pDz = mtrd.halfZ(); // Z half length
  double pDx = compWidth_/2.;
 
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD" 
		 , div_.parent().ddname().ns());
  DDSolid dsol(solname);
  DDLogicalPart ddlp(solname);
  if (!dsol.isDefined().second)
  {
    dsol = DDSolidFactory::trap(solname
				, pDz
				, 0._deg
				, 0._deg
				, pDy1
				, pDx
				, pDx
				, 0._deg
				, pDy2
				, pDx
				, pDx
				, 0._deg);
    ddlp = DDLogicalPart(solname, usemat, dsol);
  }
  return ddlp;
}

void
DDDividedTrdX::checkParametersValidity( void )
{
  DDDividedGeometryObject::checkParametersValidity();
  
  DDTrap mtrd = (DDTrap)(div_.parent().solid());

  double mpDx1 = mtrd.x1(); // X half length1
  double mpDx2 = mtrd.x2(); // X half length2
  double mpDx3 = mtrd.x3(); 
  double mpDx4 = mtrd.x4();
  double mpTheta = mtrd.theta();
  double mpPhi = mtrd.phi();
  double mpAlpha1 = mtrd.alpha1();  
  double mpAlpha2 = mtrd.alpha2();

  if ( fabs(mpDx1 - mpDx2) > tolerance()  || fabs(mpDx3 - mpDx4) > tolerance()
       || fabs(mpDx1 - mpDx4) > tolerance())
  {
    std::string s = "ERROR - DDDividedTrdX::checkParametersValidity()";
    s+= "\n        Making a division of a TRD along axis X,";
    s+= "\n        while the X half lengths are not equal,";
    s+= "\n        is not (yet) supported. It will result";
    s+= "\n        in non-equal division solids.";
    throw cms::Exception("DDException") << s;
  }

  // mec:  we only have traps, not trds in DDD, so I added this check
  // to make sure it is only a trd (I think! :-))
  if (mpAlpha1 != 0._deg || mpAlpha2 != 0._deg || mpTheta != 0._deg || mpPhi != 0._deg)
  {
    std::string s = "ERROR - DDDividedTrdX::checkParametersValidity()";
    s+= "\n        Making a division of a TRD along axis X,";
    s+= "\n        while the theta, phi and aplhpa2 are not zero,";
    s+= "\n        is not (yet) supported. It will result";
    s+= "\n        in non-equal division solids.";
    throw cms::Exception("DDException") << s;
  }
}

DDDividedTrdY::DDDividedTrdY( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionTrdY" );
  DDTrap mtrd = (DDTrap)(div_.parent().solid());

  if( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( 2 * mtrd.y1(), div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ = calculateWidth( 2 * mtrd.y1(), div_.nReplicas(), div_.offset() );
  }
}

double
DDDividedTrdY::getMaxParameter( void ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  return 2 * mtrd.y1(); 
}

DDTranslation
DDDividedTrdY::makeDDTranslation( const int copyNo ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid() );
  double mdy = mtrd.y1();

  //----- translation 
  double posi = -mdy + div_.offset() + (copyNo+0.5)*compWidth_;

  if( div_.axis() == DDAxes::y )
  {
    return DDTranslation(0.0, posi, 0.0);
  }
  else
  { 
    std::string s = "ERROR - DDDividedTrdY::makeDDTranslation()";
    s += "\n        Axis is along ";
    s += DDAxesNames::name(div_.axis());
    s += " !\n" ;
    s += "DDDividedTrdY::makeDDTranslation()";
    s += " IllegalConstruct: Only axes along y are allowed !";
    throw cms::Exception("DDException") << s;
  }
  return DDTranslation();
}

DDRotation
DDDividedTrdY::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDLogicalPart
DDDividedTrdY::makeDDLogicalPart( const int copyNo ) const
{
  //---- The division along Y of a Trd will result a Trd, only 
  //--- if Y at -Z and +Z are equal, else use the G4Trap version
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  DDMaterial usemat = div_.parent().material();
  
  double pDx1 = mtrd.x1(); // X half length1 at Y+
  double pDx2 = mtrd.x2(); // X half length2 at Y+
  double pDx3 = mtrd.x3(); // X half length1 at Y-
  double pDx4 = mtrd.x4(); // X half length2 at Y-
  double pDz = mtrd.halfZ(); // Z half length
  double pDy = compWidth_/2.;
 
  DDName solname(div_.name() );
  DDSolid  dsol(solname);
  DDLogicalPart ddlp(solname);
  if (!dsol.isDefined().second)
  {
    dsol = DDSolidFactory::trap(solname
				, pDz
				, 0._deg
				, 0._deg
				, pDy
				, pDx1
				, pDx2
				, 0._deg
				, pDy
				, pDx3
				, pDx4
				, 0._deg);
    DDLogicalPart ddlp(solname,  usemat, dsol);
  }
  return ddlp;
}

void
DDDividedTrdY::checkParametersValidity( void )
{
  DDDividedGeometryObject::checkParametersValidity();

  DDTrap mtrd = (DDTrap)(div_.parent().solid());

  double mpDy1 = mtrd.y1(); // Y half length1
  double mpDy2 = mtrd.y2(); // Y half length2
  double mpTheta = mtrd.theta();
  double mpPhi = mtrd.phi();
  double mpAlpha1 = mtrd.alpha1();
  double mpAlpha2 = mtrd.alpha2();

  if( fabs(mpDy1 - mpDy2) > tolerance() )
  {
    std::string s= "ERROR - DDDividedTrdY::checkParametersValidity()";
    s += "\n        Making a division of a TRD along axis Y while";
    s += "\n        the Y half lengths are not equal is not (yet)";
    s += "\n        supported. It will result in non-equal";
    s += "\n        division solids.";
    throw cms::Exception("DDException") << s;
  }
  // mec:  we only have traps, not trds in DDD, so I added this check
  // to make sure it is only a trd (I think! :-))
  if (mpAlpha1 != 0._deg || mpAlpha2 != 0._deg || mpTheta != 0._deg || mpPhi != 0._deg)
  {
    std::string s = "ERROR - DDDividedTrdY::checkParametersValidity()";
    s+= "\n        Making a division of a TRD along axis X,";
    s+= "\n        while the theta, phi and aplhpa2 are not zero,";
    s+= "\n        is not (yet) supported. It will result";
    s+= "\n        in non-equal division solids.";
    throw cms::Exception("DDException") << s;
  }
}

DDDividedTrdZ::DDDividedTrdZ( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject( div, cpv )
{ 
  checkParametersValidity();
  setType( "DivTrdZ" );
  DDTrap mtrd = (DDTrap)(div_.parent().solid());

  if ( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( 2*mtrd.halfZ(), div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ = calculateWidth( 2*mtrd.halfZ(), div_.nReplicas(), div_.offset() );
  }
}

double
DDDividedTrdZ::getMaxParameter( void ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  return 2 * mtrd.halfZ();
}

DDTranslation
DDDividedTrdZ::makeDDTranslation( const int copyNo ) const
{
  DDTrap mtrd = (DDTrap)(div_.parent().solid() );
  double mdz = mtrd.halfZ();

  //----- translation 
  double posi = -mdz + div_.offset() + (copyNo+0.5)*compWidth_;

  if( div_.axis() == DDAxes::z )
  {
    return DDTranslation(0.0, 0.0, posi);
  }
  else
  { 
    std::string s = "ERROR - DDDividedTrdZ::makeDDTranslation()";
    s += "\n        Axis is along ";
    s += DDAxesNames::name(div_.axis());
    s += " !\n" ;
    s += "DDDividedTrdY::makeDDTranslation()";
    s += " IllegalConstruct: Only axes along z are allowed !";
    throw cms::Exception("DDException") << s;

  }
  return DDTranslation();
}

DDRotation
DDDividedTrdZ::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDLogicalPart
DDDividedTrdZ::makeDDLogicalPart ( const int copyNo ) const
{
  //---- The division along Z of a Trd will result a Trd
  DDTrap mtrd = (DDTrap)(div_.parent().solid());
  DDMaterial usemat = div_.parent().material();

  double pDx1 = mtrd.x1(); // X half length1
  double DDx = (mtrd.x2() - mtrd.x1() );
  double pDy1 = mtrd.y1(); // Y half length1
  double DDy = (mtrd.y2() - mtrd.y1() );
  double pDz = compWidth_/2.;
  double zLength = 2*mtrd.halfZ(); // Z half length
 
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD" 
		 + std::to_string(copyNo),
		 div_.parent().ddname().ns());
  DDSolid  dsol = 
    DDSolidFactory::trap(solname
			 , pDz
			 , 0._deg
			 , 0._deg
			 , pDy1+DDy*(div_.offset()+copyNo*compWidth_)/zLength
			 , pDx1+DDx*(div_.offset()+copyNo*compWidth_)/zLength
			 , pDx1+DDx*(div_.offset()+copyNo*compWidth_)/zLength
			 , 0._deg
			 , pDy1+DDy*(div_.offset()+(copyNo+1)*compWidth_)/zLength
			 , pDx1+DDx*(div_.offset()+(copyNo+1)*compWidth_)/zLength
			 , pDx1+DDx*(div_.offset()+(copyNo+1)*compWidth_)/zLength
			 , 0._deg
      );

  DDLogicalPart ddlp(solname, usemat, dsol);
  return ddlp;
}

void
DDDividedTrdZ::checkParametersValidity( void )
{
  DDDividedGeometryObject::checkParametersValidity();

  DDTrap mtrd = (DDTrap)(div_.parent().solid());

  double mpTheta = mtrd.theta();
  double mpPhi = mtrd.phi();
  double mpAlpha1 = mtrd.alpha1();
  double mpAlpha2 = mtrd.alpha2();

  // mec:  we only have traps, not trds in DDD, so I added this check
  // to make sure it is only a trd (I think! :-))
  if (mpAlpha1 != 0._deg || mpAlpha2 != 0._deg || mpTheta != 0._deg || mpPhi != 0._deg)
  {
    std::string s = "ERROR - DDDividedTrdZ::checkParametersValidity()";
    s+= "\n        Making a division of a TRD along axis X,";
    s+= "\n        while the theta, phi and aplhpa2 are not zero,";
    s+= "\n        is not (yet) supported. It will result";
    s+= "\n        in non-equal division solids.";
    throw cms::Exception("DDException") << s;
  }
}
