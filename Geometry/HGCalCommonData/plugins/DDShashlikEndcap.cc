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
  m_zpointing   = nArgs["zpointing"];
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
  double pointingAngle = m_tiltAngle;
  double pointingLocation = m_zoffset + m_zpointing;
  double xphi = xQuadrant*(tiltAngle+pointingAngle);
  double yphi = yQuadrant*(tiltAngle+pointingAngle);
  double theta  = 90.*CLHEP::deg;
  double phiX = 0.0;
  double phiY = theta;
  double phiZ = 3*theta; 
  double offsetZ = m_zoffset;
  double offsetMultiplier = 1.00;
  //double offsetX = offsetMultiplier * offsetZ * tan( xphi - xQuadrant*pointingAngle);
  //double offsetY = offsetMultiplier * offsetZ * tan( yphi - yQuadrant*pointingAngle);
  double offsetX = offsetMultiplier * pointingLocation * tan( xphi - xQuadrant*pointingAngle );
  double offsetY = offsetMultiplier * pointingLocation * tan( yphi - yQuadrant*pointingAngle );

  int row(0), column(0);

  std::cout << " Initially, tiltAngle = " << tiltAngle 
	    << " pointingAngle=" << pointingAngle << " pointingLocation=" << pointingLocation 
	    << " copyNo = " << copyNo << ": offsetX,Y = " 
	    << offsetX << "," << offsetY 
	    << " rMin, rMax=" 
	    << m_rMin << "," << m_rMax << std::endl;

  while( abs(offsetX) < m_rMax) {
    column++;
    if (abs(offsetY) < m_rMax) 
      row++;
    while( abs(offsetY) < m_rMax) {

      double limit = sqrt( offsetX*offsetX + offsetY*offsetY );
      std::cout << " copyNo = " << copyNo << " (" << column << "," << row << "): offsetX,Y = " 
		<< offsetX << "," << offsetY << " limit=" << limit
		<< " rMin, rMax=" 
		<< m_rMin << "," << m_rMax << std::endl;

      
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
      
	std::cout << "Shashlik SM " << copyNo << ": xphi=" << xphi << " yphi=" << yphi << " offsets = (" << offsetX << ", " << offsetY << ", " << offsetZ << ")" << std::endl; 

	DDTranslation tran( offsetX, offsetY, offsetZ );
	
	DDName parentName = parent().name(); 
	cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );
	copyNo += m_incrCopyNo;
      }
      yphi += yQuadrant*2.*tiltAngle;
      //offsetY = offsetMultiplier * offsetZ * tan( yphi - yQuadrant*pointingAngle );
      offsetY = offsetMultiplier * pointingLocation * tan( yphi - yQuadrant*pointingAngle );

    }
    xphi +=  xQuadrant*2.*tiltAngle;
    yphi  =  yQuadrant*(tiltAngle + pointingAngle);
    //offsetX = offsetMultiplier * offsetZ * tan( xphi - xQuadrant*pointingAngle);
    //offsetY = offsetMultiplier * offsetZ * tan( yphi - yQuadrant*pointingAngle);
    offsetX = offsetMultiplier * pointingLocation * tan( xphi - xQuadrant*pointingAngle);
    offsetY = offsetMultiplier * pointingLocation * tan( yphi - yQuadrant*pointingAngle);




  }
  std::cout << row << " rows and " << column << " columns in quadrant " << xQuadrant << ":" << yQuadrant << std::endl;
  return copyNo;
}


