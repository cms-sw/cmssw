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
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

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
  double offsetX0 = xQuadrant*0.5*offsetXY;
  double offsetY0 = yQuadrant*0.5*offsetXY;
  
  int column = 0;
  double offsetX = offsetX0;
  while( abs(offsetX) < m_rMax)
  {
    column++;
    int row = 0;
    double offsetY = offsetY0;
    while( abs(offsetY) < m_rMax)
    {
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
      
	edm::LogInfo("HGCalGeom") << "Module " << copyNo << ":location = ("
				  << offsetX << ","
				  << offsetY << ","
				  << offsetZ << ")";

	DDTranslation tran( offsetX, offsetY, offsetZ );
	
	DDName parentName = parent().name(); 
       int absCopyNo = EKDetId::smIndex (xQuadrant>0?column:-column, yQuadrant>0?row:-row);
       cpv.position( DDName( m_childName ), parentName, absCopyNo, tran, rotation );
//      EKDetId id (absCopyNo, 13, 0, 0, 1, EKDetId::SCMODULEMODE);
//      std::cout << "quadrant " << xQuadrant<<':'<<yQuadrant<<" offset: "<<offsetX<<':'<<offsetY
//                <<" copy# " << absCopyNo
//                << " column:row " << column<<':'<<row
//                << " X:Y location "<<EKDetId::smXLocation(absCopyNo)<<':'<<EKDetId::smYLocation(absCopyNo)
//                << " ix:iy " << id.ix() << ':' << id.iy()
//                <<std::endl;

	copyNo += m_incrCopyNo;
      }

      yphi += yQuadrant*2.*tiltAngle;
      offsetY += yQuadrant*offsetXY;

      //xphi +=  xQuadrant*2.*tiltAngle;
      //// ccn: change this for no-taper
      ////offsetX = offsetZ * tan( xphi );
      //offsetX += xQuadrant*offsetXY;
    }
    xphi += xQuadrant*2.*tiltAngle;
    yphi =  yQuadrant*tiltAngle;
    // ccn: change this for no-taper
    //offsetX = offsetZ * tan( xphi );
    //offsetY = offsetZ * tan( yphi );
    offsetY = yQuadrant*0.5*offsetXY;
    offsetX += xQuadrant*offsetXY;

  }
  
  return copyNo;
}

