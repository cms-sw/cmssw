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
#include "Geometry/HGCalCommonData/plugins/DDShashlikNoTaperEndcap.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

DDShashlikNoTaperEndcap::DDShashlikNoTaperEndcap() {
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperEndcap test: Creating an instance";
}

DDShashlikNoTaperEndcap::~DDShashlikNoTaperEndcap() {}

void
DDShashlikNoTaperEndcap::initialize(const DDNumericArguments & nArgs,
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
  edm::LogInfo("SHCalGeom") << "DDShashlikNoTaperEndcap: NameSpace " << m_idNameSpace 
			    << "\tParent " << parent().name();
}

void
DDShashlikNoTaperEndcap::execute( DDCompactView& cpv )
{
  int lastCopyNo = m_startCopyNo;
  lastCopyNo = createQuarter( cpv,  1,  1, lastCopyNo );
  lastCopyNo = createQuarter( cpv, -1,  1, lastCopyNo );
  lastCopyNo = createQuarter( cpv, -1, -1, lastCopyNo );
  lastCopyNo = createQuarter( cpv,  1, -1, lastCopyNo );
}

int
DDShashlikNoTaperEndcap::createQuarter( DDCompactView& cpv, int xQuadrant, int yQuadrant, int startCopyNo )
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
  double offsetXY = m_xyoffset;

  // ccn: these need to change for no-taper option
  //double offsetX = offsetZ * tan( xphi );
  //double offsetY = offsetZ * tan( yphi );
  double offsetX = xQuadrant*0.5*offsetXY;
  double offsetY = yQuadrant*0.5*offsetXY;
  
#ifdef DebugLog
  int rowmax(0), column(0);
#endif
  while( abs(offsetX) < m_rMax) {
#ifdef DebugLog
    column++;
    int row(0);
#endif
    while( abs(offsetY) < m_rMax) {
#ifdef DebugLog
      row++;
#endif
      double limit = sqrt( offsetX*offsetX + offsetY*offsetY );
      
      // Make sure we do not add supermodules in rMin area
      if( limit > m_rMin && limit < m_rMax )
      {
#ifdef DebugLog
	std::cout << " copyNo = " << copyNo << " (" << column << "," << row 
		  << "): offsetX,Y = " << offsetX << "," << offsetY 
		  << " limit=" << limit	<< " rMin, rMax = " 
		  << m_rMin << "," << m_rMax << std::endl;
#endif
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
	edm::LogInfo("HGCalGeom") << "Module " << copyNo << ": location = "
				  << tran << " Rotation " << rotation;

	DDName parentName = parent().name(); 
	cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );

	copyNo += m_incrCopyNo;
      }

      yphi += yQuadrant*2.*tiltAngle;
      offsetY += yQuadrant*offsetXY;

      //xphi +=  xQuadrant*2.*tiltAngle;
      //// ccn: change this for no-taper
      ////offsetX = offsetZ * tan( xphi );
      //offsetX += xQuadrant*offsetXY;
    }
#ifdef DebugLog
    if (row > rowmax) rowmax = row;
#endif
    xphi += xQuadrant*2.*tiltAngle;
    yphi =  yQuadrant*tiltAngle;
    // ccn: change this for no-taper
    //offsetX = offsetZ * tan( xphi );
    //offsetY = offsetZ * tan( yphi );
    offsetY = yQuadrant*0.5*offsetXY;
    offsetX += xQuadrant*offsetXY;

  }
#ifdef DebugLog
  std::cout << rowmax << " rows and " << column << " columns in quadrant " 
	    << xQuadrant << ":" << yQuadrant << std::endl;
#endif
  return copyNo;
}

