#include "DetectorDescription/Parser/src/DDDividedPolycone.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Parser/src/DDDividedGeometryObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

class DDCompactView;

using namespace dd::operators;

DDDividedPolyconeRho::DDDividedPolyconeRho( const DDDivision& div, DDCompactView* cpv )
  :  DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();

  setType( "DivisionPolyconeRho" );

  // in DDD we only have ONE representation
  // in the case when rMinVec and rMaxVec
  // are empty rVec and zVec are the r and z std::vectors.

  if( divisionType_ == DivWIDTH )
  {
    compNDiv_ = calculateNDiv( localrMaxVec[0] - localrMinVec[0], div_.width(), div_.offset());
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ = calculateWidth( localrMaxVec[0] - localrMinVec[0], div_.nReplicas(), div_.offset());
  }
}

void
DDDividedPolyconeRho::checkParametersValidity( void )
{
  DDDividedGeometryObject::checkParametersValidity();

  DDPolycone msol = (DDPolycone)(div_.parent().solid());

  if( divisionType_ == DivNDIVandWIDTH || divisionType_ == DivWIDTH )
  {
    std::cout << "WARNING - "
	      << "DDDividedPolyconeRho::checkParametersValidity()"
	      << std::endl
	      << "          Solid " << msol << std::endl
	      << "          Division along r will be done with a width "
	      << "different for each solid section." << std::endl
	      << "          WIDTH will not be used !" << std::endl;
  }
  if( div_.offset() != 0. )
  {
    std::cout << "WARNING - "
	      << "DDDividedPolyconeRho::checkParametersValidity()"
	      << std::endl
	      << "          Solid " << msol << std::endl
	      << "          Division along  R will be done with a width "
	      << "different for each solid section." << std::endl
	      << "          OFFSET will not be used !" << std::endl;
  }
}

double
DDDividedPolyconeRho::getMaxParameter( void ) const
{
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();

  return localrMaxVec[0] - localrMinVec[0];
}

DDRotation
DDDividedPolyconeRho::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  return myddrot;
}

DDTranslation
DDDividedPolyconeRho::makeDDTranslation( const int copyNo ) const
{
  DDTranslation translation;
  return translation;
}

DDLogicalPart
DDDividedPolyconeRho::makeDDLogicalPart( const int copyNo ) const
{
  DDName solname;
  DDSolid ddpolycone;
  DDMaterial usemat(div_.parent().material());

  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();
  std::vector<double> localzVec = msol.zVec();

  int nZplanes = localzVec.size();

  std::vector<double> newrMinVec;
  std::vector<double> newrMaxVec;

  double width = 0.;
  for(int ii = 0; ii < nZplanes; ++ii)
  {
    width = calculateWidth( localrMaxVec[ii]
			    - localrMinVec[ii], compNDiv_, div_.offset() );
    // hmmm different width every time... probably should use width 
    // not compWidth_
    // 	  newrMinVec[ii] = localrMinVec[ii]+div_.offset()+compWidth_*copyNo;
    // 	  newrMaxVec[ii] = localrMinVec[ii]+div_.offset()+compWidth_*(copyNo+1);
    newrMinVec.emplace_back(localrMinVec[ii]+div_.offset()+width*copyNo);
    newrMaxVec.emplace_back(localrMinVec[ii]+div_.offset()+width*(copyNo+1));
  }
  solname = DDName( div_.parent().ddname().name() + "_DIVCHILD" + std::to_string(copyNo),
		    div_.parent().ddname().ns());
      
  ddpolycone = DDSolidFactory::polycone( solname,
					 msol.startPhi(),
					 msol.deltaPhi(),
					 localzVec,
					 newrMinVec,
					 newrMaxVec );

  DDLogicalPart ddlp = DDLogicalPart( solname, usemat, ddpolycone );
  return ddlp;
}

DDDividedPolyconePhi::DDDividedPolyconePhi( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  setType( "DivisionPolyconePhi" );
  // this is the g4.  what do we keep??? I think it is deltaPhi
  // double deltaPhi = msol->GetEndPhi() - msol->GetStartPhi();
  if( divisionType_ == DivWIDTH )
  {
    //If you divide a tube of 360 degrees the offset displaces the starting angle, but you still fill the 360 degrees
    if( msol.deltaPhi() == 360._deg ) {
      compNDiv_ = calculateNDiv( msol.deltaPhi(), div_.width(), 0. );
    }else {
      compNDiv_ = calculateNDiv( msol.deltaPhi(), div_.width(), div_.offset() );
    }
  }
  else if( divisionType_ == DivNDIV )
  {
    if( msol.deltaPhi() == 360._deg ) {
      compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), 0. );
    }else {
      compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), div_.offset() );
    }
  }
}

void
DDDividedPolyconePhi::checkParametersValidity( void )
{
  DDDividedGeometryObject::checkParametersValidity();
}

double
DDDividedPolyconePhi::getMaxParameter( void ) const
{
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  return msol.deltaPhi();
}

DDRotation
DDDividedPolyconePhi::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  double posi = ( copyNo - 1 ) * compWidth_;
  // how to name the rotation??
  // i do not like this
  DDName ddrotname( div_.parent().ddname().name() + "_DIVCHILD_ROT" + std::to_string( copyNo ),
		    div_.parent().ddname().ns());
  myddrot = DDrot( ddrotname, changeRotMatrix( posi ));

  return myddrot;
}

DDTranslation
DDDividedPolyconePhi::makeDDTranslation( const int copyNo ) const
{
  DDTranslation translation;
  return translation;
}

DDLogicalPart
DDDividedPolyconePhi::makeDDLogicalPart( const int copyNo ) const
{
  DDName solname;
  DDSolid ddpolycone;
  DDMaterial usemat(div_.parent().material());

  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();
  std::vector<double> localzVec = msol.zVec();

  solname = DDName(div_.parent().ddname().name() + "_DIVCHILD",
		   div_.parent().ddname().ns());
  DDSolid sol( solname );
  if( !sol.isDefined().second )
  {
    ddpolycone = DDSolidFactory::polycone( solname,
					   msol.startPhi()+div_.offset(),
					   compWidth_,
					   localzVec,
					   localrMinVec,
					   localrMaxVec );
  }
  DDLogicalPart ddlp( solname );
  if( !ddlp.isDefined().second )
  {
    ddlp = DDLogicalPart( solname, usemat, ddpolycone );
  }

  return ddlp;
}

DDDividedPolyconeZ::DDDividedPolyconeZ( const DDDivision& div, DDCompactView* cpv )
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();
  std::vector<double> localzVec = msol.zVec();

  if( divisionType_ == DivWIDTH )
  {
    compNDiv_ =
      calculateNDiv( localzVec[localzVec.size() - 1]
		     - localzVec[0] , div_.width(), div_.offset() );
  }
  else if( divisionType_ == DivNDIV )
  {
    compWidth_ =
      calculateNDiv( localzVec[localzVec.size()-1]
		     - localzVec[0] , div_.nReplicas(), div_.offset() );
  }
}

void
DDDividedPolyconeZ::checkParametersValidity( void )
{
  DDDividedGeometryObject::checkParametersValidity();

  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localzVec = msol.zVec();
  // CHANGE FROM G4 a polycone can be divided in Z by specifying
  // nReplicas IF they happen to coincide with the number of 
  // z plans.
  size_t tempNDiv = div_.nReplicas();
  if (tempNDiv == 0)
    tempNDiv = calculateNDiv( localzVec[localzVec.size() - 1] - localzVec[0] 
			      , div_.width()
			      , div_.offset() );
  if ((msol.zVec().size() - 1) != tempNDiv)
  { 
    std::string s = "ERROR - DDDividedPolyconeZ::checkParametersValidity()";
    s += "\n\tDivision along Z will be done splitting in the defined";
    s += "\n\tz_planes, i.e, the number of division would be :";
    s += "\n\t" + std::to_string( msol.zVec().size() - 1 );
    s += "\n\tinstead of " + std::to_string(tempNDiv) + " !\n";

    throw cms::Exception("DDException") << s;
  }
}

double
DDDividedPolyconeZ::getMaxParameter( void ) const
{
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localzVec = msol.zVec();

  return (localzVec[ localzVec.size() - 1] - localzVec[0]);
}

DDRotation
DDDividedPolyconeZ::makeDDRotation( const int copyNo ) const
{
  DDRotation myddrot; // sets to identity.
  return myddrot;
}

DDTranslation
DDDividedPolyconeZ::makeDDTranslation( const int copyNo ) const
{
  DDTranslation translation;
  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localzVec = msol.zVec();
  double posi = (localzVec[copyNo] + localzVec[copyNo+1]) / 2;
  translation.SetZ(posi);
  return translation;
}

DDLogicalPart
DDDividedPolyconeZ::makeDDLogicalPart( const int copyNo ) const
{
  DDName solname;
  DDSolid ddpolycone;
  DDMaterial usemat(div_.parent().material());

  DDPolycone msol = (DDPolycone)(div_.parent().solid());
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();
  std::vector<double> localzVec = msol.zVec();

  solname = DDName( div_.parent().ddname().name() + "_DIVCHILD" + std::to_string(copyNo),
		    div_.parent().ddname().ns());
  ddpolycone = DDSolidFactory::cons( solname,
				     compWidth_ / 2,
				     localrMinVec[copyNo],
				     localrMaxVec[copyNo],
				     localrMinVec[copyNo+1],
				     localrMaxVec[copyNo+1],
				     msol.startPhi(),
				     msol.deltaPhi());

  DDLogicalPart ddlp = DDLogicalPart( solname, usemat, ddpolycone );

  return ddlp;
}
