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
  m_xyoffset    = nArgs["xyoffset"];
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
  int lastCopyNo = m_startCopyNo;
  lastCopyNo = createQuarter( cpv,  1,  1, lastCopyNo );
  lastCopyNo = createQuarter( cpv, -1,  1, lastCopyNo );
  lastCopyNo = createQuarter( cpv, -1, -1, lastCopyNo );
  lastCopyNo = createQuarter( cpv,  1, -1, lastCopyNo );
}

int
DDShashlikEndcap::createQuarter( DDCompactView& cpv, int xQuadrant, int yQuadrant, int startCopyNo )
{
  int copyNo = startCopyNo;
  double tiltAngle = m_tiltAngle;
  double xphi = xQuadrant*tiltAngle;
  double yphi = yQuadrant*tiltAngle;
  double theta  = 90.*CLHEP::deg;
  double phiX = 0.0;
  double phiY = theta;
  double phiZ = 3*theta; 
  double offsetZ = m_zoffset;
  double offsetX = offsetZ * tan( xphi );
  double offsetY = offsetZ * tan( yphi );
  int row(0), column(0);
  while( abs(offsetX) < m_rMax ) {
    column++;
    while( abs(offsetY) < m_rMax ) {
      row++;
      double limit = sqrt( offsetX*offsetX + offsetY*offsetY );
      
      // Make sure we do not add supermodules in rMin area
      if( limit > m_rMin && limit < m_rMax )
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
	copyNo += m_incrCopyNo;
      }
      yphi += yQuadrant*2.*tiltAngle;
      offsetY = offsetZ * tan( yphi );
    }
    xphi +=  xQuadrant*2.*tiltAngle;
    yphi  =  yQuadrant*tiltAngle;
    offsetX = offsetZ * tan( xphi );
    offsetY = offsetZ * tan( yphi );
  }
  std::cout << row << " rows and " << column << " columns in quadrant " << xQuadrant << ":" << yQuadrant << std::endl;
  return copyNo;
}

