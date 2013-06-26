//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationCons*
// ********************************************************************

#include "DetectorDescription/Parser/src/DDDividedCons.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDDividedConsRho::DDDividedConsRho( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionConsRho" );
  DDCons msol = (DDCons)(div_.parent().solid());

  if( msol.rInPlusZ() == 0. )
  {
    std::cout << "WARNING - DDDividedConsRho, rOutMinusZ = 0. "
	      << std::endl 
	      << "          Width is calculated as that of rOutMinusZ !"
	      << std::endl;
  } 
      
  if( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( msol.rOutMinusZ()
			       - msol.rInMinusZ(), div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    DDCons msol = (DDCons)(div_.parent().solid());
    compWidth_ = calculateWidth( msol.rOutMinusZ() - msol.rInMinusZ()
				 , div_.nReplicas()
				 , div_.offset() );
  }

  DCOUT_V( 'P', " DDDividedConsRho - no divisions " << compNDiv_ << " = " << div_.nReplicas()
	   << "\n Offset " << div_.offset()
	   << "\n Width " << compWidth_ << " = " << div_.width()
	   << "\n DivType " << divisionType_);  
}

DDDividedConsRho::~DDDividedConsRho( void )
{}

double
DDDividedConsRho::getMaxParameter( void ) const
{
  DDCons msol = (DDCons)(div_.parent().solid());
  return msol.rOutMinusZ() - msol.rInMinusZ();

}

DDRotation
DDDividedConsRho::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  DCOUT_V ('P', "DDDividedConsRho::makeDDRotation : " << myddrot);
  return myddrot;
}

DDTranslation
DDDividedConsRho::makeDDTranslation( const int copyNo ) const
{
  //----- translation 
  DDTranslation translation;
  DCOUT_V ('P', " DDDividedConsRho " << "\n\t Position: " << translation << " - Width: " << compWidth_ << " - Axis " << DDAxesNames::name(div_.axis()));
  return translation;
}

DDLogicalPart
DDDividedConsRho::makeDDLogicalPart( const int copyNo ) const
{ 
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD" 
		 + DDXMLElement::itostr(copyNo) 
		 , div_.parent().ddname().ns());
  DDSolid ddcons(solname);
  DDMaterial usemat(div_.parent().material());
  DDCons msol = (DDCons)(div_.parent().solid());

  double pRMin1 = msol.rInMinusZ() + div_.offset() + compWidth_ * copyNo;
  double pRMax1 = msol.rInMinusZ() + div_.offset() + compWidth_ * (copyNo+1);
 
  //width at Z Plus
  //- double compWidth_Plus =
  //   compWidth_ * ( msol.rOutPlusZ()/ msol.rInPlusZ())
  //-         / ( msol.rOutMinusZ() - msol.rInMinusZ());
  double compWidth_Plus = calculateWidth( msol.rOutPlusZ()
					  - msol.rInPlusZ(), compNDiv_, div_.offset() );
  double pRMin2 = msol.rInPlusZ()
		  + div_.offset() + compWidth_Plus * copyNo;
  double pRMax2 = msol.rInPlusZ()
		  + div_.offset() + compWidth_Plus * (copyNo+1);
  double pDz = msol.zhalf();

  double pSPhi = msol.phiFrom();
  double pDPhi = msol.deltaPhi();

  ddcons = DDSolidFactory::cons(DDName(solname), pDz, pRMin1, pRMax1
				, pRMin2, pRMax2, pSPhi, pDPhi);      
  
  DDLogicalPart ddlp = DDLogicalPart(solname, usemat, ddcons);
  DCOUT_V( 'P', " DDDividedConsRho::makeDDLogicalPart() lp:" << ddlp );
  return ddlp;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
DDDividedConsPhi::DDDividedConsPhi( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionConsPhi" );
  DDCons msol = (DDCons)(div_.parent().solid());

  if( divisionType_ == DivWIDTH )
  {
    DDCons msol = (DDCons)(div_.parent().solid());
    //If you divide a tube of 360 degrees the offset displaces the starting angle, but you still fill the 360 degrees
    if( msol.deltaPhi() == 360.*deg )
    {
      compNDiv_ = calculateNDiv( msol.deltaPhi(), div_.width(), 0. );
    }
    else
    {
      compNDiv_ = calculateNDiv( msol.deltaPhi(), div_.width(), div_.offset() );
    }
  }
  else if( divisionType_ == DivNDIV )
  {
    DDCons msol = (DDCons)(div_.parent().solid());
    if( msol.deltaPhi() == 360.*deg )
    {
      compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), 0. );
    }
    else
    {
      compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), div_.offset() );
    }
  }

  DCOUT_V( 'P', " DDDividedConsPhi - no divisions " << compNDiv_ << " = " << div_.nReplicas()
	   << "\n Offset " << div_.offset()
	   << "\n Width " << compWidth_ << " = " << div_.width()
	   << "\n DivType " << divisionType_); 
}

DDDividedConsPhi::~DDDividedConsPhi( void )
{}

double
DDDividedConsPhi::getMaxParameter( void ) const
{
  DDCons msol = (DDCons)(div_.parent().solid());
  return msol.deltaPhi();
}

DDRotation
DDDividedConsPhi::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  double posi = ( copyNo - 1 ) * compWidth_;
  DDRotationMatrix* rotMat = changeRotMatrix( posi );
  // how to name the rotation??
  // i hate this crap :-)
  DDName ddrotname(div_.parent().ddname().name() + "_DIVCHILD_ROT" + DDXMLElement::itostr(copyNo)
		   , div_.parent().ddname().ns());
  myddrot = DDrot(ddrotname, rotMat);

  DCOUT_V ('P', "DDDividedConsPhi::makeDDRotation : " << myddrot);
  return myddrot;
}

DDTranslation
DDDividedConsPhi::makeDDTranslation( const int copyNo ) const
{
  //----- translation 
  DDTranslation translation;  
  DCOUT_V( 'P', " DDDividedConsPhi " << "\n\t Position: " << translation
	   << " - Width: " << compWidth_
	   << " - Axis " << DDAxesNames::name(div_.axis()));
  return translation;
}

DDLogicalPart
DDDividedConsPhi::makeDDLogicalPart( const int copyNo ) const
{ 
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD"
		 , div_.parent().ddname().ns());
  DDSolid ddcons(solname);
  DDMaterial usemat(div_.parent().material());
  DDCons msol = (DDCons)(div_.parent().solid());

  if (!ddcons.isDefined().second)
  {
    double pRMin1 = msol.rInMinusZ();
    double pRMax1 = msol.rOutMinusZ();
    double pRMin2 = msol.rInPlusZ();
    double pRMax2 = msol.rOutPlusZ();
    double pDz = msol.zhalf();
	  
    //- already rotated  double pSPhi = div_.offset() + copyNo*compWidth_;
    double pSPhi = div_.offset() + msol.phiFrom();
    double pDPhi = compWidth_;
    ddcons = DDSolidFactory::cons(DDName(solname), pDz, pRMin1, pRMax1
				  , pRMin2, pRMax2, pSPhi, pDPhi);      
  }
  
  DDLogicalPart ddlp = DDLogicalPart(solname, usemat, ddcons);
  DCOUT_V ('P', " DDDividedConsPhi::makeDDLogicalPart() lp:" << ddlp);
  return ddlp;
}

DDDividedConsZ::DDDividedConsZ( const DDDivision& div, DDCompactView* cpv )
  :  DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();

  DDCons msol = (DDCons)(div_.parent().solid());
  setType( "DivisionConsZ" );
      
  if( divisionType_ == DivWIDTH )
  {
    DDCons msol = (DDCons)(div_.parent().solid());
    compNDiv_ = calculateNDiv( 2*msol.zhalf()
			       , div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    DDCons msol = (DDCons)(div_.parent().solid());
    compWidth_ = calculateWidth( 2*msol.zhalf()
				 , div_.nReplicas(), div_.offset() );
  }

  DCOUT_V( 'P', " DDDividedConsZ - no divisions " << compNDiv_ << " = " << div_.nReplicas()
	   << "\n Offset " << div_.offset()
	   << "\n Width " << compWidth_ << " = " << div_.width()
	   << "\n DivType " << divisionType_ );
}

DDDividedConsZ::~DDDividedConsZ( void )
{}

double
DDDividedConsZ::getMaxParameter( void ) const
{
  DDCons msol = (DDCons)(div_.parent().solid());
  return 2*msol.zhalf();
}

DDRotation
DDDividedConsZ::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  DCOUT_V ('P', "DDDividedConsZ::makeDDRotation : " << myddrot);
  return myddrot;
}

DDTranslation
DDDividedConsZ::makeDDTranslation( const int copyNo ) const
{
  //----- translation 
  DDTranslation translation;

  DDCons motherCons = (DDCons)(div_.parent().solid());
  double posi = - motherCons.zhalf() + div_.offset() 
		+ compWidth_/2 + copyNo*compWidth_;
  translation.SetZ(posi); 
  
  DCOUT_V ('P', " DDDividedConsZ " << "\n\t Position: " << translation << " - Width: " << compWidth_ << " - Axis " << DDAxesNames::name(div_.axis()));
  return translation;
}

DDLogicalPart
DDDividedConsZ::makeDDLogicalPart( const int copyNo ) const
{ 
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD" + DDXMLElement::itostr(copyNo) 
		 , div_.parent().ddname().ns());
  DDSolid ddcons(solname);
  DDMaterial usemat(div_.parent().material());
  DDCons msol = (DDCons)(div_.parent().solid());

  double mHalfLength = msol.zhalf();
  double aRInner = (msol.rInPlusZ()
		    - msol.rInMinusZ()) / (2*mHalfLength);
  double bRInner = (msol.rInPlusZ()
		    + msol.rInMinusZ()) / 2;
  double aROuter = (msol.rOutPlusZ()
		    - msol.rOutMinusZ()) / (2*mHalfLength);
  double bROuter = (msol.rOutPlusZ()
		    + msol.rOutMinusZ()) / 2;
  double xMinusZ = -mHalfLength + div_.offset() + compWidth_*copyNo;
  double xPlusZ  = -mHalfLength + div_.offset() + compWidth_*(copyNo+1);

  double pDz = compWidth_ / 2.;
  double pSPhi = msol.phiFrom();
  double pDPhi = msol.deltaPhi();

  ddcons = DDSolidFactory::cons(DDName(solname)
				, pDz
				, aRInner * xMinusZ + bRInner
				, aROuter * xMinusZ + bROuter
				, aRInner * xPlusZ + bRInner
				, aROuter * xPlusZ + bROuter
				, pSPhi
				, pDPhi
    );
  
  DDLogicalPart ddlp = DDLogicalPart(solname, usemat, ddcons);
  DCOUT_V( 'P', " DDDividedConsZ::makeDDLogicalPart() lp:" << ddlp );
  return ddlp;
}
