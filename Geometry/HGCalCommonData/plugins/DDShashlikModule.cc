#include <cmath>
#include <algorithm>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikModule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikModule::DDShashlikModule() {
  edm::LogInfo("HGCalGeom") << "DDShashlikModule test: Creating an instance";
}

DDShashlikModule::~DDShashlikModule() {}

void
DDShashlikModule::initialize(const DDNumericArguments & nArgs,
			     const DDVectorArguments & vArgs,
			     const DDMapArguments & ,
			     const DDStringArguments & sArgs,
			     const DDStringVectorArguments & )
{
  m_startAngle  = nArgs["startAngle"];
  m_stepAngle   = nArgs["stepAngle"];
  m_invert      = int( nArgs["invert"] );
  m_rPos        = nArgs["rPosition"];
  m_zoffset     = nArgs["zoffset"];
  m_n           = int( nArgs["n"] );
  m_startCopyNo = int( nArgs["startCopyNo"] );
  m_incrCopyNo  = int( nArgs["incrCopyNo"] );
  m_childName   = sArgs["ChildName"];
  m_idNameSpace = DDCurrentNamespace::ns();
  edm::LogInfo("SHCalGeom") << "DDShashlikModule: NameSpace " << m_idNameSpace 
			    << "\tParent " << parent().name();
}

void
DDShashlikModule::execute( DDCompactView& cpv )
{
  double phi    = m_startAngle;
  int    copyNo = m_startCopyNo;

  for( int i = 0; i < m_n; ++i )
  {
    double phideg = phi / CLHEP::deg;
    
    DDRotation rotation;
    std::string rotstr( "NULL" );

    // Check if we've already created the rotation matrix
    rotstr = "R"; 
    rotstr  += dbl_to_string( phideg );
    rotation = DDRotation( DDName( rotstr ));
    if( !rotation )
    {
      double thetax = 90.0;
      double phix   = m_invert == 0 ? ( 90.0 + phideg ) : ( -90.0 + phideg );
      double thetay = m_invert == 0 ? 0.0 : 180.0;
      double phiz   = phideg;
      rotation = DDrot( DDName( rotstr ), thetax*CLHEP::deg, 
			phix*CLHEP::deg, thetay*CLHEP::deg, 0*CLHEP::deg,
			thetax*CLHEP::deg, phiz*CLHEP::deg);
    }

    DDTranslation tran( m_rPos * cos( phideg*CLHEP::deg), m_rPos * sin( phideg*CLHEP::deg ), m_zoffset );
    
    DDName parentName = parent().name(); 
    cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );

    phi    += m_stepAngle;
    copyNo += m_incrCopyNo;
  }
}

