#include <cmath>
#include <algorithm>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalNoTaperEndcap.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDHGCalNoTaperEndcap::DDHGCalNoTaperEndcap() {
  edm::LogInfo("HGCalGeom") << "DDHGCalNoTaperEndcap test: Creating an instance";
}

DDHGCalNoTaperEndcap::~DDHGCalNoTaperEndcap() {}

void DDHGCalNoTaperEndcap::initialize(const DDNumericArguments & nArgs,
				      const DDVectorArguments & vArgs,
				      const DDMapArguments & ,
				      const DDStringArguments & sArgs,
				      const DDStringVectorArguments & ) {
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
  DDCurrentNamespace ns;
  m_idNameSpace = *ns;
  edm::LogInfo("HGCalGeom") << "DDHGCalNoTaperEndcap: NameSpace " << m_idNameSpace 
			    << "\tParent " << parent().name();
}

void DDHGCalNoTaperEndcap::execute( DDCompactView& cpv ) {
  int lastCopyNo = m_startCopyNo;
  lastCopyNo = createQuarter( cpv,  1,  1, lastCopyNo );
  lastCopyNo = createQuarter( cpv, -1,  1, lastCopyNo );
  lastCopyNo = createQuarter( cpv, -1, -1, lastCopyNo );
  createQuarter( cpv,  1, -1, lastCopyNo );
}

int
DDHGCalNoTaperEndcap::createQuarter( DDCompactView& cpv, int xQuadrant, int yQuadrant, int startCopyNo ) {
  int copyNo       = startCopyNo;
  double tiltAngle = m_tiltAngle;
  double xphi      = xQuadrant*tiltAngle;
  double yphi      = yQuadrant*tiltAngle;
  double theta     = 90.*CLHEP::deg;
  double phiX      = 0.0;
  double phiY      = theta;
  double phiZ      = 3*theta; 
  double offsetZ   = m_zoffset;
  double offsetXY  = m_xyoffset;

  double offsetX   = xQuadrant*0.5*offsetXY;
  double offsetY   = yQuadrant*0.5*offsetXY;
  
#ifdef EDM_ML_DEBUG
  int rowmax(0), column(0);
#endif
  while (std::abs(offsetX) < m_rMax) {
#ifdef EDM_ML_DEBUG
    column++;
    int row(0);
#endif
    while (std::abs(offsetY) < m_rMax) {
#ifdef EDM_ML_DEBUG
      row++;
#endif
      double limit1 = sqrt((offsetX+0.5*xQuadrant*offsetXY)*
			   (offsetX+0.5*xQuadrant*offsetXY) + 
			   (offsetY+0.5*yQuadrant*offsetXY)*
			   (offsetY+0.5*yQuadrant*offsetXY) );
      double limit2 = sqrt((offsetX-0.5*xQuadrant*offsetXY)*
			   (offsetX-0.5*xQuadrant*offsetXY) + 
			   (offsetY-0.5*yQuadrant*offsetXY)*
			   (offsetY-0.5*yQuadrant*offsetXY) );
      // Make sure we do not add supermodules in rMin area
      if (limit2 > m_rMin && limit1 < m_rMax) {
#ifdef EDM_ML_DEBUG
	std::cout << m_childName << " copyNo = " << copyNo << " (" << column 
		  << "," << row << "): offsetX,Y = " << offsetX << "," 
		  << offsetY << " limit=" << limit1 << ":" << limit2 
		  << " rMin, rMax = " << m_rMin << "," << m_rMax << std::endl;
#endif
	DDRotation rotation;
	std::string rotstr( "NULL" );

	// Check if we've already created the rotation matrix
	rotstr   = "R"; 
	rotstr  += std::to_string(copyNo);
	rotation = DDRotation(DDName(rotstr));
	if (!rotation) {
	  rotation = DDrot(DDName(rotstr, m_idNameSpace),
			   std::make_unique<DDRotationMatrix>( *DDcreateRotationMatrix( theta, phiX, theta + yphi, phiY, -yphi, phiZ )
						 * ( *DDcreateRotationMatrix( theta + xphi, phiX, 90.*CLHEP::deg, 90.*CLHEP::deg, xphi, 0.0 ))));
	}
      

	DDTranslation tran(offsetX, offsetY, offsetZ);
	edm::LogInfo("HGCalGeom") << "Module " << copyNo << ": location = "
				  << tran << " Rotation " << rotation;

	DDName parentName = parent().name(); 
	cpv.position(DDName(m_childName), parentName, copyNo, tran, rotation);

	copyNo += m_incrCopyNo;
      } else {
#ifdef EDM_ML_DEBUG
	std::cout << " (" << column << "," << row << "): offsetX,Y = " 
		  << offsetX << "," << offsetY << " is out of limit=" << limit1
		  << ":" << limit2 << " rMin, rMax = " << m_rMin << "," 
		  << m_rMax << std::endl;
#endif
      }

      yphi += yQuadrant*2.*tiltAngle;
      offsetY += yQuadrant*offsetXY;

    }
#ifdef EDM_ML_DEBUG
    if (row > rowmax) rowmax = row;
#endif
    xphi    += xQuadrant*2.*tiltAngle;
    yphi     =  yQuadrant*tiltAngle;
    offsetY  = yQuadrant*0.5*offsetXY;
    offsetX += xQuadrant*offsetXY;

  }
#ifdef EDM_ML_DEBUG
  std::cout << rowmax << " rows and " << column << " columns in quadrant " 
	    << xQuadrant << ":" << yQuadrant << std::endl;
#endif
  return copyNo;
}

