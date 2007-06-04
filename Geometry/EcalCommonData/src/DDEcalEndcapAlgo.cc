
//////////////////////////////////////////////////////////////////////////////
// File: DDEcalEndcapAlgo.cc
// Description: Geometry factory class for Ecal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "Geometry/EcalCommonData/interface/DDEcalEndcapAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>

namespace std{} using namespace std;

DDEcalEndcapAlgo::DDEcalEndcapAlgo() :
  m_idNameSpace  ( "" ),
  m_EEName      ( "" ),
  m_EEMat       ( "" ),
  m_EEdz        (0),
  m_EErMin1     (0),
  m_EErMin2     (0),
  m_EErMax1     (0),
  m_EErMax2     (0),
  m_EEzOff      (0)
{
   edm::LogInfo("EcalGeom") << "DDEcalEndcapAlgo info: Creating an instance" ;
}

DDEcalEndcapAlgo::~DDEcalEndcapAlgo() {}




void DDEcalEndcapAlgo::initialize(const DDNumericArguments      & nArgs,
				  const DDVectorArguments       & vArgs,
				  const DDMapArguments          & mArgs,
				  const DDStringArguments       & sArgs,
				  const DDStringVectorArguments & vsArgs) {

   edm::LogInfo("EcalGeom") << "DDEcalEndcapAlgo info: Initialize" ;
   m_idNameSpace = DDCurrentNamespace::ns();
   // TRICK!
   m_idNameSpace = parent().name().ns();
   // barrel parent volume
   m_EEName     = sArgs["EEName" ] ;
   m_EEMat      = sArgs["EEMat"  ] ;
   m_EEdz       = nArgs["EEdz"  ] ;
   m_EErMin1    = nArgs["EErMin1"  ] ;
   m_EErMin2    = nArgs["EErMin2"  ] ;
   m_EErMax1    = nArgs["EErMax1"  ] ;
   m_EErMax2    = nArgs["EErMax2"  ] ;
   m_EEzOff     = nArgs["EEzOff"   ] ;
   
   edm::LogInfo("EcalGeom") << "DDEcalEndcapAlgo info: end initialize" ;
}

////////////////////////////////////////////////////////////////////
// DDEcalEndcapAlgo methods...
////////////////////////////////////////////////////////////////////

void DDEcalEndcapAlgo::execute() 
{
   edm::LogInfo("EcalGeom") << "******** DDEcalEndcapAlgo execute!" << std::endl ;

   const unsigned int copyOne (1) ;
   const unsigned int copyTwo (2) ;

   const DDSolid eeSolid ( DDSolidFactory::cons(
			      eeName() , 
			      eedz()  ,  
			      eerMin1()  ,  
			      eerMax1()  ,  
			      eerMin2()  ,  
			      eerMax2()  ,
			      0, 360*deg     ) ) ;

   const DDLogicalPart eeLog  ( eeName(), eeMat(), eeSolid ) ;
   DDpos( eeLog,
	  parent().name(), 
	  copyOne, 
	  DDTranslation(0,0, eezOff() ),
	  DDRotation() ) ;

   front() ; // do the front half including crystals

   back(); // do the back half--all passive material


   edm::LogInfo("EcalGeom") << "******** DDEcalEndcapAlgo: end it..." << std::endl ;
}

void DDEcalEndcapAlgo::front() 
{
   edm::LogInfo("EcalGeom") << "******** DDEcalEndcapAlgo front!" << std::endl ;

   edm::LogInfo("EcalGeom") << "******** DDEcalEndcapAlgo front: end it..."<< std::endl ;
}

void DDEcalEndcapAlgo::back() 
{
   edm::LogInfo("EcalGeom") << "******** DDEcalEndcapAlgo back!" << std::endl ;


   edm::LogInfo("EcalGeom") << "******** DDEcalEndcapAlgo back: end it..."<< std::endl ;
}

DDRotation
DDEcalEndcapAlgo::myrot( const std::string&      s,
			 const DDRotationMatrix& r ) const 
{
   return DDrot( ddname( m_idNameSpace + ":" + s ), new DDRotationMatrix( r ) ) ; 
}

 
DDMaterial
DDEcalEndcapAlgo::ddmat( const std::string& s ) const
{
   return DDMaterial( ddname( s ) ) ; 
}

DDName
DDEcalEndcapAlgo::ddname( const std::string& s ) const
{ 
   const pair<std::string,std::string> temp ( DDSplit(s) ) ;
   if ( temp.second == "" ) {
     return DDName( temp.first,
		    m_idNameSpace ) ;
   } else {
     return DDName( temp.first, temp.second );
   } 
}  

DDSolid    
DDEcalEndcapAlgo::mytrap( const std::string& s,
			  const EcalTrapezoidParameters& t ) const
{
   return DDSolidFactory::trap( ddname( s ),
				t.dz(), 
				t.theta(), 
				t.phi(), 
				t.h1(), 
				t.bl1(), 
				t.tl1(),
				t.alp1(), 
				t.h2(), 
				t.bl2(), 
				t.tl2(), 
				t.alp2()         ) ;
}
