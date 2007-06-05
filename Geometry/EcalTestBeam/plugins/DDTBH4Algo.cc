///////////////////////////////////////////////////////////////////////////////
// File: DDTBH4Algo.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/EcalTestBeam/plugins/DDTBH4Algo.h"
#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDTBH4Algo::DDTBH4Algo() :
   m_idNameSpace (""), 
   m_BLZBeg      (0),
   m_BLZEnd      (0),
   m_BLZPiv      (0),
   m_BLRadius    (0),
   m_VacName     (""), 
   m_VacMat      (""), 
   m_vecVacZBeg  (), 
   m_vecVacZEnd  (), 
   m_WinName    (""), 
   m_vecWinMat  (), 
   m_vecWinZBeg (), 
   m_vecWinThick(),
   m_TrgMat      (""), 
   m_HoleMat      (""), 
   m_TrgVetoHoleRadius    (0),
   m_vecTrgName  (), 
   m_vecTrgSide (), 
   m_vecTrgThick(), 
   m_vecTrgPhi  (), 
   m_vecTrgXOff (), 
   m_vecTrgYOff (), 
   m_vecTrgZPiv (),
   m_FibFibName  (""), 
   m_FibCladName (""), 
   m_FibFibMat   (""), 
   m_FibCladMat  (""), 
   m_FibSide  (0),
   m_FibCladThick(0),
   m_FibLength   (0),
   m_vecFibPhi   (), 
   m_vecFibXOff  (), 
   m_vecFibYOff  (), 
   m_vecFibZPiv  ()

{
   std::cout<< "creating an instance if DDTBH4Algo" << std::endl ;
   LogDebug("EcalGeom") << "DDTBH4Algo test: Creating an instance";
}

DDTBH4Algo::~DDTBH4Algo() {}


DDRotation
DDTBH4Algo::myrot( const std::string&      s,
		   const HepRotation& r ) const 
{
   return DDrot( ddname( idNameSpace() + ":" + s ), new DDRotationMatrix( r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), r.zx(), r.zy(), r.zz() ) ) ; 
}

 
DDMaterial
DDTBH4Algo::ddmat( const std::string& s ) const
{
   return DDMaterial( ddname( s ) ) ; 
}

DDName
DDTBH4Algo::ddname( const std::string& s ) const
{ 
   const pair<std::string,std::string> temp ( DDSplit(s) ) ;
   return DDName( temp.first,
		  temp.second ) ; 
}  


void DDTBH4Algo::initialize(const DDNumericArguments&      nArgs,
			    const DDVectorArguments&       vArgs,
			    const DDMapArguments&          mArgs,
			    const DDStringArguments&       sArgs,
			    const DDStringVectorArguments& vsArgs )
{
   m_idNameSpace = DDCurrentNamespace::ns();
   m_BLZBeg      = nArgs["BLZBeg"];	
   m_BLZEnd      = nArgs["BLZEnd"];	
   m_BLZPiv      = nArgs["BLZPiv"];	
   m_BLRadius    = nArgs["BLRadius"];	
   m_VacName     = sArgs["VacName"]; 
   m_VacMat      = sArgs["VacMat"]; 
   m_vecVacZBeg  = vArgs["VacZBeg"]; 
   m_vecVacZEnd  = vArgs["VacZEnd"]; 

   m_WinName     = sArgs["WinName"]; 
   m_vecWinMat   = vsArgs["WinMat"]; 
   m_vecWinZBeg  = vArgs["WinZBeg"]; 
   m_vecWinThick = vArgs["WinThick"]; 

   m_TrgMat       = sArgs["TrgMat"]; 
   m_HoleMat      = sArgs["HoleMat"]; 
   m_TrgVetoHoleRadius = nArgs["TrgVetoHoleRadius"];	
   m_vecTrgName   = vsArgs["TrgName"]; 
   m_vecTrgSide  = vArgs["TrgSide"]; 
   m_vecTrgThick = vArgs["TrgThick"]; 
   m_vecTrgPhi   = vArgs["TrgPhi"]; 
   m_vecTrgXOff  = vArgs["TrgXOff"]; 
   m_vecTrgYOff  = vArgs["TrgYOff"]; 
   m_vecTrgZPiv  = vArgs["TrgZPiv"]; 

   m_FibFibName  = sArgs["FibFibName"]; 
   m_FibCladName = sArgs["FibCladName"]; 
   m_FibFibMat   = sArgs["FibFibMat"]; 
   m_FibCladMat  = sArgs["FibCladMat"]; 
   m_FibSide     = nArgs["FibSide"];	
   m_FibCladThick= nArgs["FibCladThick"];	
   m_FibLength   = nArgs["FibLength"];	
   m_vecFibPhi   = vArgs["FibPhi"]; 
   m_vecFibXOff  = vArgs["FibXOff"]; 
   m_vecFibYOff  = vArgs["FibYOff"]; 
   m_vecFibZPiv  = vArgs["FibZPiv"]; 
}

void DDTBH4Algo::execute() 
{
  const unsigned int copyOne (1) ;

  const double halfZbl ( ( blZEnd() - blZBeg() )/2. ) ;
  for( unsigned int i ( 0 ) ; i != vecVacZBeg().size() ; ++i )
  {
     DDName vacNameNm ( ddname( vacName() + int_to_string(i+1) ) ) ;
     const double halfZvac ( ( vecVacZEnd()[i] - vecVacZBeg()[i] )/2. ) ;
     DDSolid vTubeSolid ( DDSolidFactory::tubs( vacNameNm ,
						halfZvac,
						0, 
						blRadius(),
						0*deg, 360*deg ) ) ;
     const DDLogicalPart vacLog ( vacNameNm, vacMat(), vTubeSolid ) ;
     
     DDpos( vacLog,
	    parent().name(), 
	    1+i, 
	    DDTranslation(0,0,
			  - halfZbl
			  + halfZvac
			  + vecVacZBeg()[i]
			  - blZBeg()),
	    DDRotation() ) ;
  }

  for( unsigned int i ( 0 ) ; i != vecWinZBeg().size() ; ++i )
  {
     DDName wName ( ddname( winName() + int_to_string(i+1)) ) ;
     DDSolid wTubeSolid ( DDSolidFactory::tubs( wName ,
						vecWinThick()[i]/2.,
						0, 
						blRadius(),
						0*deg, 360*deg ) ) ;
     const DDLogicalPart wLog ( wName, ddmat(vecWinMat()[i]), wTubeSolid ) ;

     const double off ( 0<vecWinZBeg()[i] ? vecWinZBeg()[i] :
			fabs(vecWinZBeg()[i]) - vecWinThick()[i]/2. ) ;
     
     DDpos( wLog,
	    parent().name(), 
	    1+i, 
	    DDTranslation(0,0,
			  - halfZbl
			  + vecWinThick()[i]/2.
			  + off
			  - blZBeg() ),
	    DDRotation() ) ;
  }

  for( unsigned int i ( 0 ) ; i != vecTrgName().size() ; ++i )
  {
     DDName tName ( ddname( vecTrgName()[i]) ) ;
     DDSolid tSolid ( DDSolidFactory::box( tName ,
					   vecTrgSide()[ i]/2.,
					   vecTrgSide()[ i]/2.,
					   vecTrgThick()[i]/2.  ) ) ;
     const DDLogicalPart tLog ( tName, trgMat(), tSolid ) ;

     if( tName.name() == "VETO" )
     {
	DDName vName ( ddname( tName.name() + "Hole" ) ) ;
	DDSolid vTubeSolid ( DDSolidFactory::tubs( vName ,
						   vecTrgThick()[i]/2.,
						   0, 
						   trgVetoHoleRadius(),
						   0*deg, 360*deg ) ) ;
	const DDLogicalPart vLog ( vName, holeMat(), vTubeSolid ) ;
	     
	DDpos( vLog,
	       tName, 
	       copyOne, 
	       DDTranslation(0,0,0),
	       DDRotation() ) ;
     }

     DDpos( tLog,
	    parent().name(), 
	    copyOne, 
	    DDTranslation(   vecTrgXOff()[i],
			     vecTrgYOff()[i],
			     vecTrgZPiv()[i] - halfZbl + blZPiv() - blZBeg() ),
	    myrot( tName.name()+"Rot",
		   HepRotationZ( vecTrgPhi()[i]) ) ) ;
  }

  DDName pName ( fibCladName() ) ;
  const double planeWidth ( 32.5*fibSide() +
			    33.5*fibCladThick() ) ;
  const double planeThick ( 2*fibSide() +
			    3*fibCladThick() ) ;
  DDSolid pSolid ( DDSolidFactory::box( pName ,
					planeWidth/2.,
					fibLength()/2.,
					planeThick/2. ) ) ;
  const DDLogicalPart pLog ( pName, fibCladMat(), pSolid ) ;

  DDSolid fSolid ( DDSolidFactory::box( fibFibName()  ,
					fibSide()  /2.,
					fibLength()/2.,
					fibSide()  /2.) ) ;

  const DDLogicalPart fLog ( fibFibName(), fibFibMat(), fSolid ) ;

  for( unsigned int j ( 0 ) ; j != 32 ; ++j )
  {
     const double xoff ( planeWidth/2. -
			 (1+  j)*fibCladThick() -
			 (1+2*j)*fibSide()/2.         ) ;
     const double zoff ( -planeThick/2 + fibCladThick() + fibSide()/2. ) ;
     DDpos( fLog,
	    pName, 
	    1+j, 
	    DDTranslation( xoff, 0, zoff ),
	    DDRotation() ) ;
     
     DDpos( fLog,
	    pName, 
	    33+j, 
	    DDTranslation( xoff + (fibCladThick()+fibSide())/2.,0, -zoff),
	    DDRotation() ) ;
  }
  for( unsigned int i ( 0 ) ; i != vecFibZPiv().size() ; ++i )
  {
     DDpos( pLog,
	    parent().name(), 
	    1+i, 
	    DDTranslation(   vecFibXOff()[i],
			     vecFibYOff()[i],
			     vecFibZPiv()[i] - halfZbl + blZPiv() - blZBeg() ),
	    myrot( pName.name()+"Rot" + int_to_string(i),
		   HepRotationZ( vecFibPhi()[i]) ) ) ;
  }
}
