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
#include "Geometry/HGCalCommonData/plugins/DDShashlikNoTaperSupermodule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikNoTaperSupermodule::DDShashlikNoTaperSupermodule() {
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperSupermodule test: Creating an instance";
}

DDShashlikNoTaperSupermodule::~DDShashlikNoTaperSupermodule() {}

void
DDShashlikNoTaperSupermodule::initialize(const DDNumericArguments & nArgs,
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
  edm::LogInfo("SHCalGeom") << "DDShashlikNoTaperSupermodule: NameSpace " << m_idNameSpace 
			    << "\tParent " << parent().name();
}

void
DDShashlikNoTaperSupermodule::execute( DDCompactView& cpv )
{
    int    copyNo = m_startCopyNo;


    // Variables pertaining to rotations
    // 
    //
    double phi    = m_startAngle - 2* m_stepAngle; //counting starts from edge, each module is 1*stepAngle from center, edge is 2 mods away
    double xphi   = m_startAngle - 2* m_stepAngle; //counting starts from edge, each module is 1*stepAngle from center, edge is 2 mods away
    double phideg = 0.0;
    double xphideg = 0.0;
    double theta  = 90.*CLHEP::deg;

    // these do not change module-to-module. 
    //  retaining initial values since they seem harmless
    // are these choices or necessary consequences of the parent/child frame orientations?
    double phiX = 0.0;
    double phiY = theta;
    double phiZ = 3*theta;


    // Variables pertaining to translations
    // 
    //
    // define single-module translation in xyz center-to-center (c2c), using law of sines
    //
    // check these and write new ones!...DONE
    //
    // ccn: change for no-taper  
    //double dc2c = (sin(CLHEP::pi - m_stepAngle)*(0.5*m_zoffset/sin(0.5*m_stepAngle)));
    double dc2c = m_zoffset;
    double offsetXc2c = dc2c*cos(0.5*m_stepAngle);
    double offsetYc2c = dc2c*cos(0.5*m_stepAngle);
    double offsetZc2c = dc2c*sin(0.5*m_stepAngle);

    // ccn: change for no-taper  
    //double dc2secondc = (sin(CLHEP::pi - m_stepAngle)*(dc2c/sin(0.5*m_stepAngle)));
    double dc2secondc = 2.0*m_zoffset;
    double offsetXc2secondc = dc2secondc*cos(m_stepAngle);
    double offsetYc2secondc = dc2secondc*cos(m_stepAngle);
    double offsetZc2secondc = dc2secondc*sin(m_stepAngle);

    //  starting point for two-translation scheme
    double offsetX  = -1.0 * offsetXc2secondc;
    double offsetZ1 = -1.0 * offsetZc2secondc;

    double offsetY  = -1.0 * offsetYc2secondc;
    double offsetZ2 = -1.0 * offsetZc2secondc;



//     edm::LogInfo("HGCalGeom") << "*****************";
//     edm::LogInfo("HGCalGeom") << "*****************";
//     edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperSuperModule::execute: m_startAngle = " << m_startAngle/CLHEP::deg << " deg, m_stepAngle = " << m_stepAngle/CLHEP::deg << " deg, m_zoffset = " << m_zoffset; 
//     edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperSuperModule::execute: c2c d= "<< dc2c << " = (" << offsetXc2c << "," << offsetYc2c << "," << offsetZc2c << ")"; 
//     edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperSuperModule::execute: c2secondc d= "<< dc2secondc << " = (" << offsetXc2secondc << "," << offsetYc2secondc << "," << offsetZc2secondc << ")"; 

    for( int iy = 0; iy < 5; ++iy )
    {
      
      for( int ix = 0; ix < 5; ++ix )
      {
	phideg = phi / CLHEP::deg;
	xphideg = xphi / CLHEP::deg;

	DDRotation rotation;
	std::string rotstr( "NULL" );
	std::string rotstrX( "XNULL" );

	// Check if we've already created the rotation matrix
	rotstr = "R"; 
	rotstrX = "RX"; 
	rotstr  += dbl_to_string( phideg * 10.);
	rotstrX  += dbl_to_string( xphideg * 10.);
	rotation = DDRotation( DDName( rotstr + rotstrX ));
	if( !rotation )
	{

// 	  edm::LogInfo("HGCalGeom") << "Module " << copyNo << ": first: (" 
// 	  			    << theta/CLHEP::deg << ", " 
// 	  		            << phiX/CLHEP::deg << ", "
// 	  			    << (theta+phi)/CLHEP::deg << ", " 
// 	  			    << phiY/CLHEP::deg << ", "
// 	  			    << -1.0*phi/CLHEP::deg << ", " 
// 	  			    << phiZ/CLHEP::deg << ")";
// 	  edm::LogInfo("HGCalGeom") << "Module " << copyNo << ": second: (" 
// 	  			    << (theta+xphi)/CLHEP::deg << ", " 
// 	  			    << 0.0/CLHEP::deg << ", "
// 	  			    << 90. << ", " 
// 	  			    << 90. << ", "
// 	  			    << xphi/CLHEP::deg << ", " 
// 	  			    << 0.0/CLHEP::deg << ")"; 


	  // why is phi rot in z in second matrix not phiZ instead of 0.0?
 	  rotation = DDrot( DDName( rotstr + rotstrX, m_idNameSpace ),
			    new DDRotationMatrix( *DDcreateRotationMatrix( theta, phiX, theta + phi, phiY, -phi, phiZ )
						  * ( *DDcreateRotationMatrix( theta + xphi, phiX, 90.*CLHEP::deg, 90.*CLHEP::deg, xphi, 0.0 ))));
	}

	// Translation
	//
	// we will do two translations for each module, one in xz and one in yz
	
// 	edm::LogInfo("HGCalGeom") << "Module " << copyNo << ":tran1 ("
// 				  << offsetX << ","
// 				  << 0.0 << ","
// 				  << offsetZ1 << ")";
// 	edm::LogInfo("HGCalGeom") << "Module " << copyNo << ":tran2 ("
// 				  << 0.0 << ","
// 				  << offsetY << ","
// 				  << offsetZ2 << ")";
	
// 	edm::LogInfo("HGCalGeom") << "Module " << copyNo << ":tran1+2 ("
// 				  << offsetX << ","
// 				  << offsetY << ","
// 				  << offsetZ1+offsetZ2 << ")" << " (copyNo & 5)=" << (copyNo & 5);

			  
	DDTranslation tran1( offsetX, 0.0, offsetZ1 );
 	DDTranslation tran2( 0.0, offsetY, offsetZ2 );
    
	DDName parentName = parent().name(); 
	cpv.position( DDName( m_childName ), parentName, copyNo, (tran1+tran2), rotation );



	// these are hard-coded for now
	//
	//
	if((copyNo % 5) == 1){
	  offsetZ1 = -1.0*offsetZc2c;
	  offsetX  = -1.0*offsetXc2c; 
	}else if((copyNo % 5) == 2){
	  offsetZ1 = 0.0;
	  offsetX  = 0.0; 
	}else if((copyNo % 5) == 3){
	  offsetZ1 = -1.0*offsetZc2c;
	  offsetX  = offsetXc2c; 
	}else if((copyNo % 5) == 4){
	  offsetZ1 = -1.0*offsetZc2secondc;
	  offsetX  = offsetXc2secondc; 
	}

	xphi += m_stepAngle;
	copyNo += m_incrCopyNo;

      }

      xphi    = - 2* m_stepAngle;
      phi    += m_stepAngle;
 
      offsetZ1 = -1.0*offsetZc2secondc;
      offsetX  = -1.0*offsetXc2secondc; 

      // these are hard-coded for now
      //
      //
      if(copyNo == 6){
	offsetZ2 = -1.0*offsetZc2c;
	offsetY = -1.0*offsetYc2c;
      }
      if(copyNo == 11){
	offsetZ2 = 0.0;
	offsetY = 0.0;
      }
      if(copyNo == 16){
	offsetZ2 = -1.0*offsetZc2c;
 	offsetY = offsetYc2c;
      }
      if(copyNo == 21){
	offsetZ2 = -1.0*offsetZc2secondc;
 	offsetY = offsetYc2secondc;
      }

    }

}

