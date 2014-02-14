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
DDShashlikEndcap::createQuarter( DDCompactView& cpv, int xQuadrant, int yQuadrant) {

  edm::LogInfo("HGCalGeom") << "DDShashlikEndcap::createQuarter test";

  int copyNo = m_startCopyNo;
  double phi = m_startAngle;
  double tiltAngle = m_tiltAngle;
  double xphi = 0.;
  double yphi = 0.;
  double theta  = 90.*CLHEP::deg;
  double phiX = 0.0;
  double phiY = theta;
  double phiZ = 3*theta; 
  double angle = 0.0*CLHEP::deg;
  double transX = m_rMin + m_xyoffset/2.0;
  double transY = m_xyoffset/2.0;
  double transZ = m_zoffset;
  double transLimitX = m_rMax;
  double transLimitY = m_rMax;

  if(xQuadrant==-1 && yQuadrant==1)
    copyNo += 376;
  if(xQuadrant==1 && yQuadrant==-1)
    copyNo += 752;
  if(xQuadrant==-1 && yQuadrant==-1)
    copyNo += 1128;


  for( int iy = 0; transY < transLimitY; ++iy )
  {
    for( int ix = 0; transX < transLimitX; ++ix )
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

      // Translation
      //
    //   edm::LogInfo("HGCalGeom") << "Module " << copyNo << ", quadrant (" 
// 				<< xQuadrant << "," << yQuadrant
// 				<< "), (ix,iy)=("
// 				<< ix << "," << iy
// 				<< "): transX=" << xQuadrant*transX 
// 				<< ", transY=" << yQuadrant*transY
// 				<< ", transZ=" << transZ;

   
      DDTranslation tran( xQuadrant*transX, yQuadrant*transY, transZ );
	
      DDName parentName = parent().name(); 

      cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );
      
      transX += m_xyoffset;
      if (sqrt(transX*transX + transY*transY) >= m_rMax)
	transLimitX = transX;
	
      copyNo += m_incrCopyNo;
      xphi += tiltAngle;     
    }

    if( iy < 3 )
    {
      angle = m_startAngle + m_stepAngle * iy;
      transX = (m_rMin + m_xyoffset/2.0) * cos( angle );
      ( transX < 0.0 ) ? transX = m_xyoffset/2.0 : transX;
    }
    else
      transX = m_xyoffset/2.0;

    transY += m_xyoffset;
    xphi = 0.;
    yphi += tiltAngle;
    
  }


}


void
DDShashlikEndcap::execute( DDCompactView& cpv )
{



  
  for(int i=0; i<2; i++){
    int yquad = (int)pow((-1.0),i);

    for(int j=0; j<2; j++){
 
      int xquad = (int)pow((-1.0),j);
      createQuarter(cpv, xquad, yquad);

    }
  }
 

  //createQuarter(cpv, 1, 1);


//   int copyNo = m_startCopyNo;
//   double phi = m_startAngle;
//   double tiltAngle = m_tiltAngle;
//   double xphi = 0.;
//   double yphi = 0.;
//   double theta  = 90.*CLHEP::deg;
//   double phiX = 0.0;
//   double phiY = theta;
//   double phiZ = 3*theta; 
//   double offsetX = m_rMin;
//   double offsetY = m_xyoffset;
//   double offsetZ = m_zoffset;
//   double angle = 0.0*CLHEP::deg;
//   double offsetLimitX = m_rMax;
//   double offsetLimitY = m_rMax;
    
//   for( int iy = 0; offsetY < offsetLimitY; ++iy )
//   {
//     for( int ix = 0; offsetX < offsetLimitX; ++ix )
//     {
//       DDRotation rotation;
//       std::string rotstr( "NULL" );
//       std::string rotstrX( "XNULL" );
//       std::string rotstrY( "YNULL" );

//       // Check if we've already created the rotation matrix
//       rotstr = "R"; 
//       rotstrX = "RX"; 
//       rotstrY = "RY"; 
//       rotstr  += dbl_to_string( phi * 10.);
//       rotstrX  += dbl_to_string( xphi * 10.);
//       rotstrY  += dbl_to_string( yphi * 10.);
//       rotation = DDRotation( DDName( rotstr + rotstrX + rotstrY ));
//       if( !rotation )
//       {
// 	rotation = DDrot( DDName( rotstr + rotstrX + rotstrY, m_idNameSpace ),
// 			  new DDRotationMatrix( *DDcreateRotationMatrix( theta, phiX, theta + phi, phiY, -phi, phiZ )
// 						* ( *DDcreateRotationMatrix( theta + xphi, phiX, 90.*CLHEP::deg, 90.*CLHEP::deg, xphi, 0.0 ))));
//       }

//       // Translation
//       //
//       edm::LogInfo("HGCalGeom") << "Module " << copyNo << ", (ix,iy)=("
// 				<< ix << "," << iy
// 				<< "): offsetX=" << offsetX 
// 				<< ", offsetY=" << offsetY
// 				<< ", offsetZ=" << offsetZ;
      
//       DDTranslation tran( offsetX, offsetY, offsetZ );
	
//       DDName parentName = parent().name(); 
//       cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );
      
//       offsetX += m_xyoffset;
//       if (sqrt(offsetX*offsetX + offsetY*offsetY) >= m_rMax)
// 	offsetLimitX = offsetX;
	
//       copyNo += m_incrCopyNo;
//       xphi += tiltAngle;     
//     }

//     if( iy < 3 )
//     {
//       angle = m_startAngle + m_stepAngle * iy;
//       offsetX = m_rMin * cos( angle );
//       ( offsetX < 0.0 ) ? offsetX = 0.0 : offsetX;
//     }
//     else
//       offsetX = 0.0 ;

//     offsetY += m_xyoffset;
//     xphi = 0.;
//     yphi += tiltAngle;
    
//   }


}

