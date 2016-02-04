//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationTubs*
// ********************************************************************

#include "DetectorDescription/Parser/src/DDDividedTubs.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDDividedTubsRho::DDDividedTubsRho( const DDDivision& div, DDCompactView* cpv )
  :  DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionTubsRho" );
  DDTubs msol = (DDTubs)(div_.parent().solid());

  if( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( msol.rIn() - msol.rOut(),
			       div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ = calculateWidth( msol.rIn() - msol.rOut(),
				 div_.nReplicas(), div_.offset() );
  }

  DCOUT_V ('P', " DDDividedTubsRho - no divisions " << compNDiv_ << " = " << div_.nReplicas() << "\n Offset " << div_.offset() << "\n Width " << compWidth_ << " = " << div_.width() << "\n DivType " << divisionType_);
 
}

DDDividedTubsRho::~DDDividedTubsRho( void )
{}

double
DDDividedTubsRho::getMaxParameter( void ) const
{
  DDTubs msol = (DDTubs)(div_.parent().solid());
  return msol.rOut() - msol.rIn();

}

DDRotation
DDDividedTubsRho::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  DCOUT_V ('P', "DDDividedTubsRho::makeDDRotation made a rotation: " << myddrot);
  return myddrot;
}


DDTranslation
DDDividedTubsRho::makeDDTranslation( const int copyNo ) const
{
  //----- translation 
  DDTranslation translation;

  DCOUT_V ('P', " DDDividedTubsRho " << "\n\t Position: " << translation << " - Width: " << compWidth_ << " - Axis " << DDAxesNames::name(div_.axis()));
  return translation;
}

DDLogicalPart
DDDividedTubsRho::makeDDLogicalPart( const int copyNo ) const
{
  // must always make new name and new solid
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD"
		 + DDXMLElement::itostr(copyNo) 
		 , div_.parent().ddname().ns());
  DDSolid ddtubs(solname);
  DDMaterial usemat(div_.parent().material());
  DDTubs msol = (DDTubs) (div_.parent().solid());
  DDLogicalPart ddlp;
      
  double pRMin = msol.rIn() + div_.offset() + compWidth_ * copyNo;
  double pRMax = msol.rIn() + div_.offset() + compWidth_ * (copyNo+1);
  double pDz = msol.zhalf();
  double pSPhi = msol.startPhi();
  double pDPhi = msol.deltaPhi();
  ddtubs = DDSolidFactory::tubs(DDName(solname), pDz, pRMin, pRMax, pSPhi, pDPhi);      
  ddlp = DDLogicalPart(solname, usemat, ddtubs);

  DCOUT_V ('P', " DDDividedTubsRho::computeDimensions() lp:" << ddlp);
  return ddlp;
}

DDDividedTubsPhi::DDDividedTubsPhi( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionTubsPhi" );

  DDTubs msol = (DDTubs)(div_.parent().solid());
  if( divisionType_ == DivWIDTH )
  {
    //If you divide a tube of 360 degrees the offset displaces the starting angle, but you still fill the 360 degrees
    if( msol.deltaPhi() == 360.*deg ) {
      compNDiv_ = calculateNDiv( msol.deltaPhi(), div_.width(), 0. );
    }else {
      compNDiv_ = calculateNDiv( msol.deltaPhi(), div_.width(), div_.offset() );
    }

  }
  else if( divisionType_ == DivNDIV )
  {
    if( msol.deltaPhi() == 360.*deg ) {
      compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), 0. );
    }else {
      compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), div_.offset() );
    }
  }
}

DDDividedTubsPhi::~DDDividedTubsPhi( void )
{}

double
DDDividedTubsPhi::getMaxParameter( void ) const
{
  DDTubs msol = (DDTubs)(div_.parent().solid());
  return msol.deltaPhi();
}

DDRotation
DDDividedTubsPhi::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  double posi = ( copyNo - 1 ) * compWidth_; // This should put the first one at the 0 of the parent.
  DDRotationMatrix * rotMat = changeRotMatrix( posi );
  // how to name the rotation??
  // i hate this crap :-)
  DDName ddrotname(div_.parent().ddname().name() + "_DIVCHILD_ROT"
		   + DDXMLElement::itostr(copyNo)
		   , div_.parent().ddname().ns());
  myddrot = DDrot(ddrotname, rotMat);

  return myddrot;
}

DDTranslation
DDDividedTubsPhi::makeDDTranslation( const int copyNo ) const
{
  //----- translation 
  DDTranslation translation;

  DCOUT_V ('P', " DDDividedTubsPhi " << "\n\t Position: " << translation << " - Width: " << compWidth_ << " - Axis " << DDAxesNames::name(div_.axis()));
  return translation;
}

DDLogicalPart
DDDividedTubsPhi::makeDDLogicalPart( const int copyNo ) const
{
  DDName solname(div_.name());
  //-  DDName solname(div_.parent().ddname().name() + "_DIVCHILD", div_.parent().ddname().ns());
  DDSolid ddtubs(solname);
  DDMaterial usemat(div_.parent().material());
  DDTubs msol = (DDTubs) (div_.parent().solid());
  DDLogicalPart ddlp(solname);

  if (!ddtubs.isDefined().second)  // only if it is not defined, make new dimensions and solid.
  {
    double pRMin = msol.rIn();
    double pRMax = msol.rOut();
    double pDz = msol.zhalf();
    double pSPhi = msol.startPhi()+div_.offset(); 
    double pDPhi = compWidth_;  
    ddtubs = DDSolidFactory::tubs(DDName(solname), pDz, pRMin, pRMax, pSPhi, pDPhi);
    ddlp = DDLogicalPart(solname, usemat, ddtubs);
  }

  DCOUT_V ('P', " DDDividedTubsPhi::computeDimensions() lp:" << ddlp);
  return ddlp;
}

DDDividedTubsZ::DDDividedTubsZ( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();

  DDTubs msol = (DDTubs)(div_.parent().solid());

  setType( "DivisionTubsZ" );
  if( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( 2*msol.zhalf(), div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ = calculateWidth( 2*msol.zhalf(), div_.nReplicas(), div_.offset() );
  }

  DCOUT_V ('P', " DDDividedTubsZ - no divisions " << compNDiv_ << " = " << div_.nReplicas() << "\n Offset " << div_.offset() << "\n Width " << compWidth_ << " = " << div_.width() << "\n DivType " << divisionType_);
 
}

DDDividedTubsZ::~DDDividedTubsZ( void )
{}

double
DDDividedTubsZ::getMaxParameter( void ) const
{
  DDTubs msol = (DDTubs)(div_.parent().solid());
  return 2*msol.zhalf();

}

DDRotation
DDDividedTubsZ::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  DCOUT_V ('P', "DDDividedTubsZ::makeDDRotation made a rotation: " << myddrot);
  return myddrot;
}

DDTranslation
DDDividedTubsZ::makeDDTranslation( const int copyNo ) const
{
  //----- translation 
  DDTranslation translation;

  DDTubs msol = (DDTubs)(div_.parent().solid());
  double posi = - msol.zhalf() + div_.offset() + compWidth_/2 + copyNo*compWidth_;
  translation.SetZ(posi);
  
  DCOUT_V ('P', " DDDividedTubsZ " << "\n\t Position: " << translation << " - Width: " << compWidth_ << " - Axis " << DDAxesNames::name(div_.axis()));
  return translation;
}

DDLogicalPart
DDDividedTubsZ::makeDDLogicalPart( const int copyNo ) const
{
  DDMaterial usemat(div_.parent().material());
  DDTubs msol = (DDTubs) (div_.parent().solid());
  DDLogicalPart ddlp;

  DDName solname(div_.parent().ddname().name() + "_DIVCHILD", div_.parent().ddname().ns());
  DDSolid ddtubs(solname);

  if (!ddtubs.isDefined().second)  // only if it is not defined, make new dimensions and solid.
  {
    double pRMin = msol.rIn();
    double pRMax = msol.rOut();
    double pDz = compWidth_/2.;
    double pSPhi = msol.startPhi();
    double pDPhi = msol.deltaPhi();
    ddtubs = DDSolidFactory::tubs(DDName(solname), pDz, pRMin, pRMax, pSPhi, pDPhi);
    ddlp = DDLogicalPart(solname, usemat, ddtubs);
  }
  else {
    ddlp = DDLogicalPart(solname);
  }

  DCOUT_V ('P', " DDDividedTubsZ::computeDimensions() lp:" << ddlp);
  return ddlp;
}
