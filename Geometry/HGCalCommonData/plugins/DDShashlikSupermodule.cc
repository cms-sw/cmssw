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
#include "Geometry/HGCalCommonData/plugins/DDShashlikSupermodule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikSupermodule::DDShashlikSupermodule() {
  edm::LogInfo("HGCalGeom") << "DDShashlikSupermodule test: Creating an instance";
}

DDShashlikSupermodule::~DDShashlikSupermodule() {}

void
DDShashlikSupermodule::initialize(const DDNumericArguments & nArgs,
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
  edm::LogInfo("SHCalGeom") << "DDShashlikSupermodule: NameSpace " << m_idNameSpace 
			    << "\tParent " << parent().name();
}

void
DDShashlikSupermodule::execute( DDCompactView& cpv )
{
  double phi    = m_startAngle;
  //int    copyNo = m_startCopyNo;
  double phideg = 0.0;

  /*
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

    DDTranslation tran( (m_rPos * cos( phideg*CLHEP::deg)), (i*7.0+m_rPos * sin( phideg*CLHEP::deg )), m_zoffset );
    
    DDName parentName = parent().name(); 
    cpv.position( DDName( m_childName ), parentName, copyNo, tran, rotation );

    phi    += m_stepAngle;
    copyNo += m_incrCopyNo;
  }
  */



  // Brute force

  // x24 x23 x22 x21 x20    ^ y
  // x19 x18 x17 x16 x15    |
  // x14 x13 x12 x11 x10    |
  //  x9  x8  x7  x6 x5     -----> z
  //  x4  x3  x2  x1 x0

  //ccn: change these to access widthFront and widthBack and Thickness
  double PI = 4.0 * atan(1.0);
  double TWOPI = 2.0*PI;
  long double MODLENGTH = 113.5;
  long double MODINNERSIDE = 14.0;
  long double MODMIDSIDE = 14.28;
  long double MODOUTERSIDE = 14.56;

  long double taper_angle = atan((MODOUTERSIDE-MODINNERSIDE)*0.5 / MODLENGTH);
  long double taper_angle_deg = taper_angle / CLHEP::deg;
 edm::LogInfo("HGCalGeom") << "DDShashlikSuperModule:: taper_angle = " << taper_angle << " rad (" << taper_angle_deg << " deg)"; 

  long double f2f_rotation = 2.0*taper_angle;
  //long double f2f_rotation = taper_angle;
  long double f2f_rotation_deg = f2f_rotation / CLHEP::deg;
  edm::LogInfo("HGCalGeom") << "DDShashlikSuperModule:: f2f_rotation = " << f2f_rotation << " rad (" << f2f_rotation_deg << " deg)"; 

  long double f2f_angle = TWOPI - (PI + 2.0*taper_angle);
  long double f2f_angle_deg = f2f_angle / CLHEP::deg;
  edm::LogInfo("HGCalGeom") << "DDShashlikSuperModule:: f2f_angle = " << f2f_angle << " rad (" << f2f_angle_deg << " deg)"; 

  // law of sines to get distance from face-to-face
  long double f2f_dist = sin(f2f_angle)*(0.5*MODINNERSIDE/sin(taper_angle));
  edm::LogInfo("HGCalGeom") << "DDShashlikSuperModule:: f2f_dist = " << f2f_dist << " mm";

  long double c2c_dist = sin(f2f_angle)*(0.5*MODMIDSIDE/sin(taper_angle));
  edm::LogInfo("HGCalGeom") << "DDShashlikSuperModule:: c2c_dist = " << c2c_dist << " mm";


  // module 12
  //
  phi = 0.0;
  phideg = phi / CLHEP::deg;
    
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

  // ccn: m_rPos = 0.0 and m_zoffset = 0.0 here, so these thre args are each 0.0
  DDTranslation tran( (m_rPos * cos( phideg*CLHEP::deg)), (m_rPos * sin( phideg*CLHEP::deg )), m_zoffset );
    
  DDName parentName = parent().name(); 
  cpv.position( DDName( m_childName ), parentName, 12, tran, rotation );


  
  // module 17
  //
  phideg = f2f_rotation_deg;    
  //phideg = -2.0;

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
  
  tran = DDTranslation( (-1.0*c2c_dist*sin(taper_angle) + m_rPos * cos( phideg*CLHEP::deg)), (c2c_dist*cos(taper_angle) + m_rPos * sin( phideg*CLHEP::deg )), (m_zoffset) );

  //tran = DDTranslation( (-1.0*c2c_dist*sin(taper_angle)), (c2c_dist*cos(taper_angle)),0.0 );

  parentName = parent().name(); 
  cpv.position( DDName( m_childName ), parentName, 17, tran, rotation );
  

}

