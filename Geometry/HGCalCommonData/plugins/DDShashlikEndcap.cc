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
#include "Geometry/HGCalCommonData/plugins/DDShashlikEndcap.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikEndcap::DDShashlikEndcap() {
  edm::LogInfo("HGCalGeom") << "DDShashlikEndcap test: Creating an instance";
}

DDShashlikEndcap::~DDShashlikEndcap() {}

void
DDShashlikEndcap::initialize(const DDNumericArguments & nArgs,
			     const DDVectorArguments & vArgs,
			     const DDMapArguments & ,
			     const DDStringArguments & sArgs,
			     const DDStringVectorArguments & )
{
  m_startAngle  = nArgs["startAngle"];
  m_stepAngle   = nArgs["stepAngle"];
  m_tiltAngle   = nArgs["tiltAngle"];
  m_invert      = int( nArgs["invert"] );
  m_rMin        = int( nArgs["rMin"] );
  m_rMax        = int( nArgs["rMax"] );
  m_rPos        = nArgs["rPosition"];
  m_xyoffset    = nArgs["xyoffset"];
  m_zoffset     = nArgs["zoffset"];
  m_n           = int( nArgs["n"] );
  m_startCopyNo = int( nArgs["startCopyNo"] );
  m_incrCopyNo  = int( nArgs["incrCopyNo"] );
  m_childName   = sArgs["ChildName"];
  m_idNameSpace = DDCurrentNamespace::ns();
  edm::LogInfo("SHCalGeom") << "DDShashlikEndcap: NameSpace " << m_idNameSpace 
			    << "\tParent " << parent().name();
}

void
DDShashlikEndcap::execute( DDCompactView& cpv )
{
  int copyNo = m_startCopyNo;
  double phi = m_startAngle;
  double tiltAngle = m_tiltAngle;
  double xphi = 0.;
  double yphi = 0.;
  double theta  = 90.*CLHEP::deg;
  double phiX = 0.0;
  double phiY = theta;
  double phiZ = 3*theta; 
  double offsetX = m_rMin;
  double offsetY = m_xyoffset;
  double offsetZ = m_zoffset;
  double angle = 0.0*CLHEP::deg;
  double offsetLimitX = m_rMax;
  double offsetLimitY = m_rMax;
    
  for( int iy = 0; offsetY < offsetLimitY; ++iy )
  {
    for( int ix = 0; offsetX < offsetLimitX; ++ix )
    {
      DDRotation rotation;
      std::string rotstr( "NULL" );
      std::string rotstrX( "XNULL" );
      std::string rotstrY( "YNULL" );

      // Check if we've already created the rotation matrix
      rotstr = "R"; 
      rotstrX = "RX"; 
      rotstrY = "RY"; 
      rotstr  += dbl_to_string( phi * 10.);
      rotstrX  += dbl_to_string( xphi * 10.);
      rotstrY  += dbl_to_string( yphi * 10.);
      rotation = DDRotation( DDName( rotstr + rotstrX + rotstrY ));
      if( !rotation )
      {
	rotation = DDrot( DDName( rotstr + rotstrX + rotstrY, m_idNameSpace ),
			  new DDRotationMatrix( *DDcreateRotationMatrix( theta, phiX, theta + phi, phiY, -phi, phiZ )
						* ( *DDcreateRotationMatrix( theta + xphi, phiX, 90.*CLHEP::deg, 90.*CLHEP::deg, xphi, 0.0 ))));
      }
      
      DDTranslation tran( offsetX, offsetY, offsetZ );
	
      DDName parentName = parent().name(); 
      cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );
      
      offsetX += m_xyoffset;
      if (sqrt(offsetX*offsetX + offsetY*offsetY) >= m_rMax)
	offsetLimitX = offsetX;
	
      copyNo += m_incrCopyNo;
      xphi += tiltAngle;     
    }

    if( iy < 3 )
    {
      angle = m_startAngle + m_stepAngle * iy;
      offsetX = m_rMin * cos( angle );
      ( offsetX < 0.0 ) ? offsetX = 0.0 : offsetX;
    }
    else
      offsetX = 0.0 ;

    offsetY += m_xyoffset;
    xphi = 0.;
    yphi += tiltAngle;
    
  }
}

