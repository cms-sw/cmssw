//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationPolyhedra*
//---------------------------------------------------------------------

#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/Parser/interface/DDDividedPolyhedra.h"
#include "DetectorDescription/Parser/interface/DDXMLElement.h"

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

//--------------------------------------------------------------------------
DDDividedPolyhedraRho::DDDividedPolyhedraRho( const DDDivision & div )
  :  DDDividedGeometryObject( div )
{
  checkParametersValidity();
  setType( "DivisionPolyhedraRho" );

  DDPolyhedra msol = (DDPolyhedra)( div_.parent().solid() );
  // G4PolyhedraHistorical* original_pars = msol->GetOriginalParameters();

  if( divisionType_ == DivWIDTH )
    {
      compNDiv_ = calculateNDiv( msol.rMaxVec()[0] - msol.rMinVec()[0]
				 , div_.width()
				 , div_.offset() );
    }
  else if( divisionType_ == DivNDIV )
    {
      compWidth_ = calculateWidth( msol.rMaxVec()[0] - msol.rMinVec()[0]
				   , div_.nReplicas()
				   , div_.offset() );
    }

  //     for (int i = 0; i < compNDiv_; ++i)
  //      {
  //        DDpos(  makeDDLogicalPart(i)
  //  	      , div_.parent()
  //  	      , i
  //  	      , makeDDTranslation(i)
  //  	      , makeDDRotation(i)
  //  	      , &div_
  //  	      );
  //      } 
 
  DCOUT_V ('P', " DDDividedPolyhedraRho - # divisions " << compNDiv_  << " = " << div_.nReplicas() << "\n Offset " << div_.offset() << " Width " << compWidth_ << " = " << div_.width() << "\n");

}

//------------------------------------------------------------------------
DDDividedPolyhedraRho::~DDDividedPolyhedraRho()
{
}

//---------------------------------------------------------------------
void DDDividedPolyhedraRho::checkParametersValidity()
{
  DDDividedGeometryObject::checkParametersValidity();

  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());

  if( divisionType_ == DivNDIVandWIDTH || divisionType_ == DivWIDTH )
    {
      std::cout << "WARNING - "
	   << "DDDividedPolyhedraRho::checkParametersValidity()"
	   << std::endl
	   << "          Solid " << msol << std::endl
	   << "          Division along R will be done with a width "
	   << "different for each solid section." << std::endl
	   << "          WIDTH will not be used !" << std::endl;
    }
  if( div_.offset() != 0. )
    {
      std::cout << "WARNING - "
	   << "DDDividedPolyhedraRho::checkParametersValidity()"
	   << std::endl
	   << "          Solid " << msol << std::endl
	   << "          Division along  R will be done with a width "
	   << "different for each solid section." << std::endl
	   << "          OFFSET will not be used !" << std::endl;
    }


}

//------------------------------------------------------------------------
double DDDividedPolyhedraRho::getMaxParameter() const
{
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  //  G4PolyhedraHistorical* original_pars = msol->GetOriginalParameters();
  return msol.rMaxVec()[0] - msol.rMinVec()[0];
}

//--------------------------------------------------------------------------
DDTranslation DDDividedPolyhedraRho::makeDDTranslation( const int copyNo ) const
{
  return DDTranslation();
}

//--------------------------------------------------------------------------
DDRotation DDDividedPolyhedraRho::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

//--------------------------------------------------------------------------
DDLogicalPart DDDividedPolyhedraRho::makeDDLogicalPart ( const int copyNo ) const
{
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  DDMaterial usemat = div_.parent().material();
  //G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  //G4PolyhedraHistorical origparam( *origparamMother );
  std::vector<double> localrMaxVec = msol.rMaxVec();
  std::vector<double> localrMinVec = msol.rMinVec();
  std::vector<double> localzVec = msol.zVec(); 
  std::vector<double> newrMinVec;
  std::vector<double> newrMaxVec;
  int nZplanes = localzVec.size();

  double width = 0.;
  for(int ii = 0; ii < nZplanes; ++ii)
    {
      //     width = CalculateWidth( origparamMother->Rmax[ii]
      //                           - origparamMother->Rmin[ii], compNDiv_, foffset );
      //     origparam.Rmin[ii] = origparamMother->Rmin[ii]+foffset+width*copyNo;
      //     origparam.Rmax[ii] = origparamMother->Rmin[ii]+foffset+width*(copyNo+1);
      width = calculateWidth(localrMaxVec[ii] - localrMinVec[ii], compNDiv_, div_.offset());
      newrMinVec[ii] = localrMinVec[ii] + div_.offset() + width * copyNo;
      newrMaxVec[ii] = localrMaxVec[ii] + div_.offset() + width * (copyNo + 1);
    }

  //   phedra.SetOriginalParameters(&origparam); // copy values & transfer pointers
  //   phedra.Reset();                           // reset to new solid parameters
  
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD" + DDXMLElement::itostr(copyNo) 
		 , div_.parent().ddname().ns());
  
  DDSolid dsol = DDSolidFactory::polyhedra(solname
					   , msol.sides()
					   , msol.startPhi()
					   , msol.deltaPhi()
					   , localzVec
					   , newrMinVec
					   , newrMaxVec);
  DDLogicalPart ddlp = DDLogicalPart(solname, usemat, dsol);
  DCOUT_V ('P', "DDDividedPolyhedraRho:makeDDLogicalPart lp:" <<  ddlp);
  return ddlp;
}

//--------------------------------------------------------------------------
DDDividedPolyhedraPhi::DDDividedPolyhedraPhi ( const DDDivision & div )
  :  DDDividedGeometryObject( div )
{ 
  checkParametersValidity();
  setType( "DivisionPolyhedraPhi" );

  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  //  double deltaPhi = msol->GetEndPhi() - msol->GetStartPhi();
  
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
	// original line looks wrong!
	compWidth_ = calculateWidth( msol.deltaPhi(), div_.nReplicas(), div_.offset() );
      }
    }
 
  DCOUT_V ('P', " DDDividedPolyhedraRho - # divisions " << compNDiv_  << " = " << div_.nReplicas() << "\n Offset " << div_.offset() << " Width " << compWidth_ << " = " << div_.width() << "\n");
}

//------------------------------------------------------------------------
DDDividedPolyhedraPhi::~DDDividedPolyhedraPhi()
{
}

//------------------------------------------------------------------------
double DDDividedPolyhedraPhi::getMaxParameter() const
{
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  return msol.deltaPhi(); //msol->GetEndPhi() - msol->GetStartPhi();
}

//---------------------------------------------------------------------
void DDDividedPolyhedraPhi::checkParametersValidity()
{
  DDDividedGeometryObject::checkParametersValidity();
  
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  
  if( divisionType_ == DivNDIVandWIDTH || divisionType_ == DivWIDTH )
    {
      std::cout << "WARNING - "
           << "DDDividedPolyhedraPhi::checkParametersValidity()"
           << std::endl
           << "          Solid " << msol << std::endl
           << "          Division along PHI will be done splitting "
           << "in the defined numSide." << std::endl
           << "          WIDTH will not be used !" << std::endl;
    }
  if( div_.offset() != 0. )
    {
      std::cout << "WARNING - "
           << "DDDividedPolyhedraPhi::checkParametersValidity()"
           << std::endl
           << "          Solid " << msol << std::endl
           << "          Division along PHI will be done splitting "
           << "in the defined numSide." << std::endl
           << "          OFFSET will not be used !" << std::endl;
    }
  
  //  G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  
  if ( msol.sides() != compNDiv_ )//origparamMother->numSide != compNDiv_ )
    { 
      std::cout << "ERROR - "
	   << "DDDividedPolyhedraPhi::checkParametersValidity()"
	   << std::endl
	   << "        Division along PHI will be done splitting in the defined"
	   << std::endl
	   << "        numSide, i.e, the number of division would be :"
	   << "        " << msol.sides() //origparamMother->numSide
	   << " instead of " << compNDiv_ << " !"
	   << std::endl; 
      std::string s = "DDDividedPolyhedraPhi::checkParametersValidity() Not supported configuration.";
      throw DDException(s);
    }
}


//--------------------------------------------------------------------------
DDTranslation DDDividedPolyhedraPhi::makeDDTranslation( const int copyNo ) const
{
  return DDTranslation();
}

//--------------------------------------------------------------------------
DDRotation DDDividedPolyhedraPhi::makeDDRotation( const int copyNo ) const
{

  double posi = ( copyNo - 1 ) * compWidth_;

  DCOUT_V ('P', " DDDividedPolyhedraPhi - position: " << posi/deg << "\n copyNo: " << copyNo << " - compWidth_: " << compWidth_/deg << "\n");
  
  //  ChangeRotMatrix( physVol, -posi );
  DDRotationMatrix* rotMat = changeRotMatrix( posi);
  // how to name the rotation??
  // i do not like this...
  DDName ddrotname(div_.parent().ddname().name() + "_DIVCHILD_ROT" + DDXMLElement::itostr(copyNo)
		   , div_.parent().ddname().ns());
  DDRotation myddrot = DDrot(ddrotname, rotMat);
  DCOUT_V ('P', "DDDividedPolyhedra::makeDDRotation: copyNo = " << copyNo << " rotation = " << myddrot);
  return myddrot;

}

//--------------------------------------------------------------------------
DDLogicalPart DDDividedPolyhedraPhi::makeDDLogicalPart( const int copyNo ) const
{
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  DDMaterial usemat = div_.parent().material();
  //G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  //G4PolyhedraHistorical origparam( *origparamMother );

  //origparam.numSide = 1;
  //origparam.Start_angle = origparamMother->Start_angle;
  //origparam.Opening_angle = compWidth_;

  //phedra.SetOriginalParameters(&origparam);  // copy values & transfer pointers
  //phedra.Reset();                            // reset to new solid parameters

  DDName solname(div_.parent().ddname().name() + "_DIVCHILD"
		 , div_.parent().ddname().ns());
  DDSolid dsol(solname);
  if (!dsol.isDefined().second)
    {
      dsol = DDSolidFactory::polyhedra(solname
				       , msol.sides()
				       , msol.startPhi()+div_.offset()
				       , compWidth_
				       , msol.zVec()
				       , msol.rMinVec()
				       , msol.rMaxVec()
				       );
    }
  DDLogicalPart ddlp(solname, usemat, dsol);
  DCOUT_V ('P', "DDDividedPolyhedraPhi::makeDDLogicalPart() ddlp = " << ddlp);
  return ddlp;
}

//--------------------------------------------------------------------------
DDDividedPolyhedraZ::DDDividedPolyhedraZ ( const DDDivision & div )
  :  DDDividedGeometryObject ( div )
{ 
  checkParametersValidity();
  setType( "DivisionPolyhedraZ" );
  
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  //G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  std::vector<double> zvec = msol.zVec();
  
  if  ( divisionType_ == DivWIDTH )
    {
      compNDiv_ =
	calculateNDiv( zvec[zvec.size() - 1] - zvec[0], div_.width(), div_.offset() );
    }
  else if( divisionType_ == DivNDIV )
    {
      compWidth_ = calculateWidth( zvec[zvec.size() - 1] - zvec[0]
				   , div_.nReplicas()
				   , div_.offset());
      // ?what?      CalculateNDiv( zvec[zvec.size() - 1] - zvec[0], origparamMother->Z_values[origparamMother->Num_z_planes-1]
      //       - origparamMother->Z_values[0] , nDiv, offset );
      
    }
  
  DCOUT_V ('P', " DDDividedPolyhedraZ - # divisions " << compNDiv_ << " = " << div_.nReplicas() << "\n Offset " << " = " << div_.offset() << "\n Width " << compWidth_ << " = " << div_.width());
  
}
					   
//---------------------------------------------------------------------
DDDividedPolyhedraZ::~DDDividedPolyhedraZ()
{
}

//------------------------------------------------------------------------
double DDDividedPolyhedraZ::getMaxParameter() const
{
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  //G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  std::vector<double> zvec = msol.zVec();
  return (zvec[zvec.size() - 1] - zvec[0]);
  //origparamMother->Z_values[origparamMother->Num_z_planes-1]
  //	  -origparamMother->Z_values[0]);
}

//---------------------------------------------------------------------
void DDDividedPolyhedraZ::checkParametersValidity()
{
  DDDividedGeometryObject::checkParametersValidity();

  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());

  if( divisionType_ == DivNDIVandWIDTH || divisionType_ == DivWIDTH )
    {
      std::cout << "WARNING - "
           << "DDDividedPolyhedraZ::checkParametersValidity()"
           << std::endl
           << "          Solid " << msol << std::endl
           << "          Division along Z will be done splitting "
           << "in the defined z_planes." << std::endl
           << "          WIDTH will not be used !" << std::endl;
    }

  if( div_.offset() != 0. )
    {
      std::cout << "WARNING - "
           << "DDDividedPolyhedraZ::checkParametersValidity()"
           << std::endl
           << "          Solid " << msol << std::endl
           << "          Division along Z will be done splitting "
           << "in the defined z_planes." << std::endl
           << "          OFFSET will not be used !" << std::endl;
    }

  //  G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();

  std::vector<double> zvec = msol.zVec();
  
  if ( zvec.size() - 1 != size_t(compNDiv_) )
    { 
      std::cout << "ERROR - "
	   << "DDDividedPolyhedraZ::checkParametersValidity()"
	   << std::endl
	   << "        Division along Z can only be done by splitting in the defined"
	   << std::endl
	   << "        z_planes, i.e, the number of division would be :"
	   << "        " << zvec.size() - 1
	   << " instead of " << compNDiv_ << " !"
	   << std::endl; 
      std::string s = "DDDividedPolyhedraZ::checkParametersValidity()";
      s += "Illegal Construct. Not a supported configuration.";
      throw DDException (s);
    }
}

//---------------------------------------------------------------------
DDTranslation DDDividedPolyhedraZ::makeDDTranslation( const int copyNo ) const
{
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  std::vector<double> zvec = msol.zVec();
  
  //----- set translation: along Z axis
  // G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  //    double posi = (origparamMother->Z_values[copyNo]
  //                     + origparamMother->Z_values[copyNo+1])/2;
  
  double posi = (zvec[copyNo] + zvec[copyNo+1])/2;

  //G4ThreeVector origin(0.,0.,posi); 
  //  physVol->SetTranslation( origin );
  
  DDTranslation tr(0,0,posi);
  //----- calculate rotation matrix: unit
  
  DCOUT_V ('P', " DDDividedPolyhedraZ - position: " << posi << "\n copyNo: " << copyNo << " - offset: " << div_.offset()/deg << " - compWidth_: " << compWidth_/deg << " translation = " << tr);
  return tr;
}

//---------------------------------------------------------------------
DDRotation DDDividedPolyhedraZ::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

//---------------------------------------------------------------------
DDLogicalPart DDDividedPolyhedraZ::makeDDLogicalPart( const int copyNo ) const
{
  // only for mother number of planes = 2!!
  // mec: what?  why?  comment above and = 2 below straight from G4 impl.
  DDPolyhedra msol = (DDPolyhedra)(div_.parent().solid());
  DDMaterial usemat = div_.parent().material();
  //    G4PolyhedraHistorical* origparamMother = msol->GetOriginalParameters();
  //    G4PolyhedraHistorical origparam( *origparamMother );
  std::vector<double> zvec = msol.zVec();
  std::vector<double> rminvec = msol.rMinVec();
  std::vector<double> rmaxvec = msol.rMaxVec();

  double posi = (zvec[copyNo] + zvec[copyNo+1])/2;
  
  //    origparam.Num_z_planes = 2;
  //    origparam.Z_values[0] = origparamMother->Z_values[copyNo] - posi;
  //    origparam.Z_values[1] = origparamMother->Z_values[copyNo+1] - posi;
  //    origparam.Rmin[0] = origparamMother->Rmin[copyNo];
  //    origparam.Rmin[1] = origparamMother->Rmin[copyNo+1];
  //    origparam.Rmax[0] = origparamMother->Rmax[copyNo];
  //    origparam.Rmax[1] = origparamMother->Rmax[copyNo+1];
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD" 
		 + DDXMLElement::itostr(copyNo)
		 , div_.parent().ddname().ns());
  std::vector<double> newRmin, newRmax, newZ;
  newZ.push_back(zvec[copyNo] - posi);
  newZ.push_back(zvec[copyNo+1] - posi);
  newRmin.push_back(rminvec[copyNo]);
  newRmin.push_back(rminvec[copyNo+1]);
  newRmax.push_back(rmaxvec[copyNo]);
  newRmax.push_back(rmaxvec[copyNo+1]);

  DDSolid dsol = DDSolidFactory::polyhedra(solname
					   , msol.sides()
					   , msol.startPhi()
					   , msol.deltaPhi()
					   , newZ
					   , newRmin
					   , newRmax);
  DDLogicalPart lp(solname, usemat, dsol);

  //    phedra.SetOriginalParameters(&origparam);  // copy values & transfer pointers
  //    phedra.Reset();                            // reset to new solid parameters

  DCOUT_V ('P', "DDDividedPolyhedraZ::makeDDLogicalPart" << "\n-- Parametrised phedra copy-number: " << copyNo << "\n-- DDLogicalPart " << lp );
  return lp;

}

