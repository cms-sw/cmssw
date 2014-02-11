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
  m_tiltAngle   = nArgs["tiltAngle"];
  m_invert      = int( nArgs["invert"] );
  m_rMin        = int( nArgs["rMin"] );
  m_rMax        = int( nArgs["rMax"] );
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
  double tiltAngle = m_tiltAngle;
  double xphi = tiltAngle;
  double yphi = tiltAngle;
  double theta  = 90.*CLHEP::deg;
  double phiX = 0.0;
  double phiY = theta;
  double phiZ = 3*theta; 
  double offsetZ = m_zoffset;
  double offsetX = offsetZ * sin( 2 * xphi );
  double offsetY = offsetZ * sin( 2 * yphi );
  double offsetLimitX = m_rMax;
  double offsetLimitY = m_rMax;

  for( int iy = 0; offsetY < offsetLimitY; ++iy, yphi += 2*tiltAngle )
  {
    xphi = tiltAngle;
    offsetX = offsetZ * sin( xphi );
    offsetY = offsetZ * sin( yphi );

    for( int ix = 0; offsetX < offsetLimitX; ++ix, xphi += 2*tiltAngle )
    {
      offsetX = offsetZ * sin( xphi );
      
      // Make sure we do not add supermodules in rMin area
      if( sqrt( offsetX*offsetX + offsetY*offsetY ) > m_rMin )
      {
	DDRotation rotation;
	std::string rotstr( "NULL" );

	// Check if we've already created the rotation matrix
	rotstr = "R"; 
	rotstr  += dbl_to_string( copyNo );
	rotation = DDRotation( DDName( rotstr ));
	if( !rotation )
	{
	  rotation = DDrot( DDName( rotstr, m_idNameSpace ),
			    new DDRotationMatrix( *DDcreateRotationMatrix( theta, phiX, theta + yphi, phiY, -yphi, phiZ )
						  * ( *DDcreateRotationMatrix( theta + xphi, phiX, 90.*CLHEP::deg, 90.*CLHEP::deg, xphi, 0.0 ))));
	}
      
	DDTranslation tran( offsetX, offsetY, offsetZ );
	
	DDName parentName = parent().name(); 
	cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );

	if (sqrt(offsetX*offsetX + offsetY*offsetY) >= m_rMax)
	  offsetLimitX = offsetX;
	
	copyNo += m_incrCopyNo;
      }
    }
  }
}

