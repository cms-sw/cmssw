//   COCOA class implementation file
//Id:  OptOCops.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOCOPS.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Model.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "Alignment/CocoaModel/interface/ALILine.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>     		// I have added a new library for edm::isNotFinite() function
#include <cstdlib>

using namespace CLHEP;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as distance to previous object 'screen'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCOPS::defaultBehaviour( LightRay& lightray, Measurement& meas )
{
  if(ALIUtils::debug >= 5) std::cout << "***** OptOCOPS::defaultBehaviour" <<std::endl;
  makeMeasurement( lightray, meas);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as distance to previous object 'screen'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCOPS::makeMeasurement( LightRay& lightray, Measurement& meas ) 
{

  if (ALIUtils::debug >= 4) std::cout << "***** OptOCOPS::makeMeasurement(lightray, meas) " << std::endl; 
  //---------- Centre of COPS is at dowel point 2 
  CLHEP::Hep3Vector dowel2 = centreGlob();
  //---------- Coordinates of dowel point 1 are given with respect to dowel point 2 (in local reference frame)
  ALIdouble posx12 = findExtraEntryValue("dowel1X");
// Changed default value to .045 from .03
  if(posx12 == 0. ) posx12 = -0.045; //samir changed sign to correct the dowel 1st pixel
  CLHEP::Hep3Vector dowel1(posx12,findExtraEntryValue("dowel1Y"), 0.);
  CLHEP::HepRotation rmt = rmGlob(); 
  dowel1 = rmt*dowel1;
  dowel1 += dowel2;  
  if (ALIUtils::debug >= 3) {
    ALIUtils::dump3v(dowel2, " dowel2");
    ALIUtils::dump3v(dowel1, " dowel1");
  }

  //---------- Get line joining dowel1-dowel2 and perpendicular to it inside cops
//  CLHEP::Hep3Vector line_dowel21 = - (dowel1-dowel2 ); //////
  CLHEP::Hep3Vector line_dowel21 =  (dowel1-dowel2 ); // samir changed sign to correct the dowel 1st pixel
  CLHEP::Hep3Vector ZAxis(0.,0,1.); 
  ZAxis = rmt * ZAxis; 
  CLHEP::Hep3Vector line_dowel21_perp = ZAxis.cross( line_dowel21 ); 
  if (ALIUtils::debug >= 3) {
    ALIUtils::dump3v(line_dowel21," line_dowel21");
    ALIUtils::dump3v(line_dowel21_perp," line_dowel21_perp");
  }
 
  //---------- Position four CCDs (locally, i.e. with respect to centre)
  //----- Get first CCD length, that will be used to give a default placement to the CCDs
  ALIdouble CCDlength = findExtraEntryValue("CCDlength");
  if( CCDlength == 0. ) CCDlength = 2048*14 * 1.E-6; // (in meters, the default unit)



  //  global / local output of ccd location in RF was reversed, I am swapping


  //----- Upper CCD (leftmost point & direction dowel1-dowel2)
  if(ALIUtils::debug>= 3) std::cout << std::endl << "***** UP CCD *****" << std::endl
                            << "******************" << std::endl << std::endl;
  ALIdouble posX = findExtraEntryValue("upCCDXtoDowel2");
  ALIdouble posY;
  ALIbool eexists = findExtraEntryValueIfExists("upCCDYtoDowel2", posY);
  if(!eexists) posY = CCDlength + 0.004;
  //if(!eexists) posY =  0.004;
  CLHEP::Hep3Vector posxy( posX, posY, 0);
  if(ALIUtils::debug>= 3) std::cout << " %%%% CCD distances to Dowel2: " << std::endl;
  if(ALIUtils::debug>= 3) std::cout << "   up ccd in local RF " << posxy << std::endl; 
  posxy = rmt * posxy;
  if(ALIUtils::debug>= 3) std::cout << "   up ccd in global RF " << posxy << std::endl; 
 // ALILine upCCD( dowel2 + posxy, -line_dowel21 ); 
 // ccds[0] = ALILine( posxy, -line_dowel21 );
   ALILine upCCD( dowel2 + posxy, line_dowel21 );// Samir changed sign to correct the dowel 1st pixel
  ccds[0] = ALILine( posxy, line_dowel21 ); // samir changed sign to correct the dowel 1st pixel
  //----- Lower CCD (leftmost point & direction dowel2-dowel1)
  if(ALIUtils::debug>= 3) std::cout << std::endl << "***** DOWN CCD *****" << std::endl 
                            << "********************" << std::endl << std::endl ;
  posX = findExtraEntryValue("downCCDXtoDowel2");
  eexists = findExtraEntryValueIfExists("downCCDYtoDowel2", posY);
  if(!eexists) posY = 0.002;
  posxy = CLHEP::Hep3Vector( posX, posY, 0);
  if(ALIUtils::debug>= 3) std::cout << "   down ccd in local RF " << posxy << std::endl; 
  posxy = rmt * posxy;
  if(ALIUtils::debug>= 3) std::cout << "   down ccd in global RF " << posxy << std::endl; 
//  ALILine downCCD( dowel2 + posxy, -line_dowel21 );
 // ccds[1] = ALILine( posxy, -line_dowel21 );

ALILine downCCD( dowel2 + posxy, line_dowel21 );//samir changed signto correct the dowel 1st pixel
  ccds[1] = ALILine( posxy, line_dowel21 ); // samir changed sign to correct the dowel 1st pixel

  //----- left CCD (uppermost point & direction perpendicular to dowel2-dowel1)

  if(ALIUtils::debug>= 3) std::cout << std::endl << "***** LEFT CCD *****" << std::endl
                            << "********************" << std::endl << std::endl;
  eexists = findExtraEntryValueIfExists("leftCCDXtoDowel2", posX);
//  if(!eexists) posX = -0.002;
  if(!eexists) posX = -CCDlength - 0.002; // Samir changed sign to correct the dowel 1st pixel
  posY = findExtraEntryValue("leftCCDYtoDowel2");
  posxy = CLHEP::Hep3Vector( posX, posY, 0);
  if(ALIUtils::debug>= 3) std::cout << "   left ccd in local RF " << posxy << std::endl; 
  posxy = rmt * posxy;
  if(ALIUtils::debug>= 3) std::cout << "   left ccd in global RF " << posxy << std::endl; 
 // ALILine leftCCD( dowel2 + posxy, line_dowel21_perp );
 // ccds[2] = ALILine(  posxy, line_dowel21_perp );

  ALILine leftCCD( dowel2 + posxy, -line_dowel21_perp );//samir changed sign to correct the dowel 1st pixel
  ccds[2] = ALILine(  posxy, -line_dowel21_perp );//samir changed sign to correct the dowel 1st pixel

  //----- right CCD (uppermost point & direction perpendicular to dowel2-dowel1)
  if(ALIUtils::debug>= 3) std::cout << std::endl << "***** RIGHT CCD *****" << std::endl 
                            << "*********************" << std::endl<< std::endl ;
  eexists = findExtraEntryValueIfExists("rightCCDXtoDowel2", posX);
 // if(!eexists) posX = -CCDlength - 0.004;
  if(!eexists) posX =  - 0.004; // samir tried to change in order to adjust the position of 1 st pixel.
  posY = findExtraEntryValue("rightCCDYtoDowel2");
  posxy = CLHEP::Hep3Vector( posX, posY, 0);
  if(ALIUtils::debug>= 3) std::cout << "   right ccd in local RF " << posxy << std::endl; 
  posxy = rmt * posxy;
  if(ALIUtils::debug>= 3) std::cout << "   right ccd in global RF " << posxy  << std::endl << std::endl; 
 // ALILine rightCCD( dowel2 + posxy, line_dowel21_perp );
 // ccds[3] = ALILine(  posxy, line_dowel21_perp );
  
ALILine rightCCD( dowel2 + posxy, -line_dowel21_perp ); //samir changed sign to correct the dowel 1st pixel
  ccds[3] = ALILine(  posxy, -line_dowel21_perp ); //samir changed sign to correct the dowel 1st pixel

  if (ALIUtils::debug >= 3) {
    std::cout << " %%%  Positions of CCDs in global RF: " << std::endl<< std::endl;
    std::cout << "     upCCD: " << upCCD << std::endl;
    std::cout << "     downCCD: " << downCCD << std::endl;
    std::cout << "     leftCCD: " << leftCCD << std::endl;
    std::cout << "     rightCCD: " << rightCCD << std::endl << std::endl;
  }

  //---------- Intersect x-hair laser with COPS
  if (ALIUtils::debug >= 3) std::cout << " %%%  Intersecting x-hair laser with COPS: " << std::endl;
  ALIPlane copsPlane(centreGlob(), ZAxis);
  lightray.intersect( *this ); 
  CLHEP::Hep3Vector inters = lightray.point();
  if (ALIUtils::debug >= 3) {
     ALIUtils::dump3v(inters, " Intersection of x-hair laser with COPS ");
  }

  //---------- Get cross of x-hair laser:
   if (ALIUtils::debug >= 5) std::cout << "1. Get the OptO x-hair laser from the measurement list of OptOs" << std::endl;
   
  OpticalObject* xhairOptO = *(meas.OptOList().begin());

  if (ALIUtils::debug >= 35) std::cout << "2. Get the Y of the laser and project it on the COPS" << std::endl;
  CLHEP::Hep3Vector YAxis_xhair(0.,1.,0.);
  CLHEP::HepRotation rmtx = xhairOptO->rmGlob(); 
  YAxis_xhair = rmtx * YAxis_xhair;
  ALILine Yline_xhair( inters, copsPlane.project( YAxis_xhair ));
  if (ALIUtils::debug >= 3) {
    std::cout << "  %%%% Projecting x-hair laser on COPS: " << std::endl;
     ALIUtils::dump3v(YAxis_xhair, " Y direction of laser ");
    std::cout << " Y line of laser projected on COPS " <<  Yline_xhair << std::endl;
  }

  if (ALIUtils::debug >= 5) std::cout << " 3. Get the X of the laser (correct it if cross is not 90o) and project it on the COPS" << std::endl;
  
  ALIdouble anglebx;
  eexists = xhairOptO->findExtraEntryValueIfExists("angleBetweenAxis", anglebx);
  if(!eexists) anglebx = PI/2.;
  CLHEP::Hep3Vector XAxis_xhair = YAxis_xhair;

 //   if (ALIUtils::debug >= 3) ALIUtils::dump3v(XAxis_xhair," X of laser1 ");
  ZAxis = CLHEP::Hep3Vector(0.,0.,1.);
  ZAxis = rmtx * ZAxis;
  XAxis_xhair.rotate(anglebx, ZAxis );
  ALILine Xline_xhair( inters, copsPlane.project( XAxis_xhair ) );
  if (ALIUtils::debug >= 3) {
    std::cout << "angleBetweenAxis = " << anglebx << std::endl;
    ALIUtils::dump3v(XAxis_xhair," X direction of laser ");
    std::cout << " X line of laser projected on COPS " <<  Xline_xhair << std::endl;
  }
 

  //---------- Get measurement as intersection with four CCDs 
  if(ALIUtils::debug >= 3) std::cout << "  Getting measurements as intersection with four CCDs: " << std::endl;

  if(ALIUtils::debug >= 4)std::cout << "intersecting with upCCD " << std::endl;
  ALIdouble measv[4][2];

// swap Y and X line_xhair by exchanging second index

  if(ALIUtils::debug >= 5)std::cout << "$@S@ measv[0][0] upccd " << std::endl;
  measv[0][0] = getMeasFromInters( Yline_xhair, upCCD, line_dowel21 );
  if(ALIUtils::debug >= 5)std::cout << "$@$@ measv[0][1] upccd " << std::endl;
  measv[0][1] = getMeasFromInters( Xline_xhair, upCCD, line_dowel21 );
  
  //---- check if postive or negative: 
  if(ALIUtils::debug >= 4) std::cout << "intersecting with downCCD " << std::endl;
  measv[1][0] = getMeasFromInters(Yline_xhair, downCCD, line_dowel21 );
  measv[1][1] = getMeasFromInters(Xline_xhair, downCCD, line_dowel21 );
  
//  
  
  if(ALIUtils::debug >= 4) std::cout << "intersecting with leftCCD " << std::endl;
  measv[2][0] = getMeasFromInters(Xline_xhair, leftCCD, line_dowel21_perp );
  measv[2][1] = getMeasFromInters(Yline_xhair, leftCCD, line_dowel21_perp );
  if(ALIUtils::debug >= 4) std::cout << "intersecting with rightCCD " << std::endl;
  measv[3][0] = getMeasFromInters(Xline_xhair, rightCCD, line_dowel21_perp );
  measv[3][1] = getMeasFromInters(Yline_xhair, rightCCD, line_dowel21_perp );

  /* Both X and Y axis of the x-laser are intersected with each CCD and it checks that one of 
  the two is inside the CCD(less than CCDlength/2). If no one is inside, it will give an 
  exception. If both are inside (a strange case where, for example, the laser centre is very 
  close and forms 45 degrees with the CCD) it will also make an exception (if you prefer, I can
  put a warning saying that you have two measurements, but I guess this should never happen for
  you, so I better give an exception and you don't risk to overpass this warning).

  Then it is mandatory that you put the CCDlength parameter (I could put a default one if 
  you prefer). 
  

  ALIbool measInCCD[2];
  ALIuint ii,jj;
  for( ii = 0; ii < 4; ii++ ) {
    for( jj = 0; jj < 2; jj++ ) {
      measInCCD[jj] =  fabs( measv[ii][jj] ) < CCDlength/2;
    }
    if (ALIUtils::debug >= 2) std::cout << "$@$@ CHECK CCD = " << ii << std::endl; 
    if( measInCCD[0] && measInCCD[1] ){
      std::cerr << "!!!EXITING: both lasers lines of x-hair laser intersect with same CCD " << measNames[ii] << " one at " << measv[ii][0] << " another one at " << measv[ii][1] << "CCDhalfLegth " << CCDlength/2 << std::endl;
      exit(1);
    } else if( !(measInCCD[0] || measInCCD[1]) ){
      std::cerr << "!!!EXITING: none of the lasers lines of x-hair laser intersect with CCD " << measNames[ii] << ", one at " << measv[ii][0] << " another one at " << measv[ii][1] << "CCDhalfLegth " << CCDlength/2 << std::endl;
      exit(1);
    } else {
      measInCCD[0] ?  meas.setValueSimulated( ii, measv[ii][0] ) :
	meas.setValueSimulated( ii, measv[ii][1] );
    }
  }
  */

  ALIstring measNames[4] ={"up","down","left","right"};
  ALIbool laserLine; 
  if (ALIUtils::debug >= 2) std::cout << std::endl << "--> Now comparing measurement in ccds by x and y laser lines (will always choose the smaller one) " <<std::endl; 
     
  unsigned int ii;
  for( ii = 0; ii < 4; ii++ ) {
    if (ALIUtils::debug >= 2) std::cout << "\tmeas CCD " << measNames[ii] << " ii=(" << ii << ") \t Values: "
     //<< (fabs( measv[ii][0] ) <  fabs( measv[ii][1]) 
       << " " << fabs( measv[ii][0] ) << " " <<  fabs( measv[ii][1] ) << "  edm::isNotFinite() = " <<
       edm::isNotFinite(measv[ii][1]) << std::endl;

    if( meas.xlaserLine( ii ) != -1 ) { 
      laserLine = ALIbool( meas.xlaserLine( ii ) );
    } else { 
    
    //  Problem here !!!
    //
    //  Somehow measv[][1] can occasionally return value of 'nan'
    //  which is interpretted as less than any real value
    //
      if(edm::isNotFinite(measv[ii][1]) != 0){
      		measv[ii][1] = 1e99;
		if (ALIUtils::debug >= 2) std::cout << "  --> Swapping for " << measv[ii][1] << "(inf)" << std::endl;
				  }
				   
      laserLine = fabs( measv[ii][0] ) <  fabs( measv[ii][1] );
      
      meas.setXlaserLine( ii, int(laserLine) );
    }
    laserLine ? meas.setValueSimulated( ii, measv[ii][0] ) 
      : meas.setValueSimulated( ii, measv[ii][1] );
  }
  
    if (ALIUtils::debug >= 2) std::cout << std::endl; 	//Keep format of debug output reasonable

  // try to identify pathological cases: up and down CCD are intersected by the same 
  // laser line (say X) and the same for the left and right CCD

  if(ALIUtils::debug >= 2) std::cout << "***** OptOCOPS::makeMeasurement  - identify pathological cases U and D intersected by same line" <<std::endl;
  ALIbool xlaserDir[4];
  for( ii = 0; ii < 4; ii++ ) {
    //    xlaserDir[ii] = fabs( measv[ii][0] ) <  fabs( measv[ii][1] );
    xlaserDir[ii] = ALIbool( meas.xlaserLine( ii ) ); 
  }
  if( xlaserDir[0] ^ xlaserDir[1] ) {
    std::cerr << "!!EXITING up and down CCDs intersected by different x-laser line " << xlaserDir[0] << " " <<  xlaserDir[1] << std::endl;
    exit(1);
  }
  if( xlaserDir[2] ^ xlaserDir[3] ) {
    std::cerr << "!!EXITING right and left CCDs intersected by different x-laser line " << xlaserDir[0] << " " <<  xlaserDir[1] << std::endl;
    exit(1);
  }

  if(ALIUtils::debug >= 5) std::cout << "***** OptOCOPS::makeMeasurement - now output sim values" << std::endl;

  if (ALIUtils::debug >= 1) {
    ALIstring chrg = "";
    std::cout << "REAL value: " << chrg <<"U: " << 1000*meas.value()[0] << chrg 
	 << " D: " << 1000*meas.value()[1] 
	 << " L: " << 1000*meas.value()[2] 
	 << " R: " << 1000*meas.value()[3]  
	 << " (mm)  " << (this)->name() << std::endl;
    ALIdouble detU =  1000*meas.valueSimulated(0); if(fabs(detU) <= 1.e-9 ) detU = 0.;
    ALIdouble detD =  1000*meas.valueSimulated(1); if(fabs(detD) <= 1.e-9 ) detD = 0.;
    ALIdouble detL =  1000*meas.valueSimulated(2); if(fabs(detL) <= 1.e-9 ) detL = 0.;
    ALIdouble detR =  1000*meas.valueSimulated(3); if(fabs(detR) <= 1.e-9 ) detR = 0.;
    std::cout << "SIMU value: " << chrg << "U: "
      // << setprecision(3) << setw(4)
	 << detU
	 << chrg << " D: " << detD
	 << chrg << " L: " << detL
	 << chrg << " R: " << detR
	 << " (mm)  " << (this)->name() << std::endl;
  }
  

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of Light Ray traverses
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCOPS::fastTraversesLightRay( LightRay& lightray )
{

  if(ALIUtils::debug >= 5) std::cout << "***** OptOCOPS::fastTraversesLightRay" <<std::endl;
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST TRAVERSES COPS  " << name() << std::endl;

  //---------- Get intersection 
  CLHEP::Hep3Vector ZAxis(0.,0,1.);
  CLHEP::HepRotation rmt = rmGlob();
  ZAxis = rmt * ZAxis;
  lightray.intersect( ALIPlane(centreGlob(), ZAxis) );
  CLHEP::Hep3Vector inters = lightray.point();
  lightray.setPoint( inters );

  if (ALIUtils::debug >= 2) {
    lightray.dumpData(" after COPS ");
  }

}     


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble* OptOCOPS::convertPointToLocalCoordinates( const CLHEP::Hep3Vector& point)
{
  if(ALIUtils::debug >= 1) std::cout << "***** OptOCOPS::convertPointToLocalCoordinates" <<std::endl;
  ALIdouble* interslc = new ALIdouble[2];

  //----- X value
  CLHEP::HepRotation rmt = rmGlob();
  CLHEP::Hep3Vector XAxism(1.,0.,0.);
  XAxism*=rmt;
  if( ALIUtils::debug >= 5) ALIUtils::dump3v( (this)->centreGlob(),  "centre glob sensor2D" );
  if( ALIUtils::debug >= 5) ALIUtils::dumprm( rmt,  "rotation matrix sensor2D" );
  //t  ALIUtils::dump3v(point - (this)->centreGlob() , "inters - (this)->centreGlob()");
  //t  ALIUtils::dump3v(XAxism , "XAxism");
  interslc[0] = (point - (this)->centreGlob() ) * XAxism;
  
  //----- Y value
  CLHEP::Hep3Vector YAxism(0.,1.,0.);
  YAxism*=rmt;
  //t  ALIUtils::dump3v(YAxism , "YAxism");
  interslc[1] = (point - (this)->centreGlob() ) * YAxism;

  if( ALIUtils::debug >=5 ) {
    std::cout << " intersection in local coordinates: X= " << interslc[0] << "  Y= " << interslc[1] << std::endl;
  }
  return interslc;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble OptOCOPS::getMeasFromInters( ALILine& line_xhair, ALILine& ccd, CLHEP::Hep3Vector& cops_line )
{

  if(ALIUtils::debug >= 5) std::cout << "***** OptOCOPS::getMeasFromInters" <<std::endl;
  CLHEP::Hep3Vector inters = line_xhair.intersect( ccd, 0 ) - ccd.pt(); 
  ALIdouble sign = inters*ccd.vec(); 
  if( sign != 0 ){
    sign = fabs(sign)/sign;
    //    std::cout << "  @@@@@@@@@@@@ sign = " << sign << std::endl;
    //   ALIUtils::dump3v(inters, " intersection " );
    //   ALIUtils::dump3v( ccd.vec(), " cops_line ");
  } //sign can be zero only if inters is 0, because they are parallel
  return sign*inters.mag();
}


#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCOPS::fillVRML()
{
 
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();

  //---------- Position four CCDs (locally, i.e. with respect to centre)
  //----- Get first CCD length, that will be used to give a default placement to the CCDs
  ALIdouble CCDlength = findExtraEntryValue("CCDlength");
  if( CCDlength == 0. ) CCDlength = 2048*14 * 1.E-6; // (in meters, the default unit)
  //Original CCDdim value was 10; .05 works well
//  ALIdouble CCDdim = .040;
    ALIdouble CCDdim = 2048*14 * 1.E-6;
  //Original was divided by 10
//  ALIdouble CCDwidth = CCDdim/10.;
  ALIdouble CCDwidth = .005;

  if(ALIUtils::debug >= 4) std::cout << " ccds0 " << ccds[0] << "ccds1 " << ccds[1] << std::endl;
  ALIColour colup( 1.,0.,0., 0.);
  ALIColour coldown( 0.,1.,0., 0.);
  ALIColour colleft( 0.,0.,1., 0.);
  ALIColour colright( 0.,1.,1., 0.);
  //----- Upper CCD (leftmost point & direction dowel1-dowel2)
  //original ccdsize was 50; 1 works ok
  ALIdouble ccdsize = 1.;

  // VRML objects are drawn 'after rotation into system'; so to make all CCDs look the
  //  same in the drawing, CCDwidth parameters must be the same
  // x, y , z , color

  //----- Upper CCD 
  vrmlmgr.AddBoxDisplaced( *this, CCDdim, CCDwidth, CCDwidth, ccdsize*(ccds[0].pt()+0.5*CCDlength*ccds[0].vec()), &colup);
  //----- Lower CCD (leftmost point & direction dowel2-dowel1)
  vrmlmgr.AddBoxDisplaced( *this, CCDdim, CCDwidth, CCDwidth, ccdsize*(ccds[1].pt()+0.5*CCDlength*ccds[1].vec()), &coldown );
  //----- left CCD (uppermost point & direction perpendicular to dowel2-dowel1)
  vrmlmgr.AddBoxDisplaced( *this, CCDwidth, CCDdim, CCDwidth, ccdsize*(ccds[2].pt()+0.5*CCDlength*ccds[2].vec()), &colleft );
  //----- right CCD (uppermost point & direction perpendicular to dowel2-dowel1)
  vrmlmgr.AddBoxDisplaced( *this, CCDwidth, CCDdim, CCDwidth, ccdsize*(ccds[3].pt()+0.5*CCDlength*ccds[3].vec()), &colright );

  vrmlmgr.SendReferenceFrame( *this, CCDdim); 
  vrmlmgr.SendName( *this, 0.01 );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCOPS::fillIguana()
{
  //---------- Position four CCDs (locally, i.e. with respect to centre)
  //----- Get first CCD length, that will be used to give a default placement to the CCDs
  ALIdouble CCDlength;
  ALIbool pexists = findExtraEntryValueIfExists("CCDlength",CCDlength);
  if( !pexists ) CCDlength = 2048*14 * 1.E-6;
  ALIdouble CCDdim = 2048*14 * 1.E-6;
  ALIdouble CCDwidth = .005;

  if(ALIUtils::debug >= 4) std::cout << " ccds0 " << ccds[0] << "ccds1 " << ccds[1] << std::endl;
  ALIColour colup( 0.2,1.,0., 0.);
  ALIColour coldown( 0.3,1.,0., 0.);
  ALIColour colleft( 0.4,1.,0., 0.);
  ALIColour colright( 0.5,1.,0., 0.);
  //----- Upper CCD (leftmost point & direction dowel1-dowel2)
  //original ccdsize was 50; 1 works ok
  ALIdouble ccdsize = 1;

  // VRML objects are drawn 'after rotation into system'; so to make all CCDs look the
  //  same in the drawing, CCDwidth parameters must be the same
  // x, y , z , color

  std::vector<ALIdouble> spar;
  spar.push_back(CCDdim);
  spar.push_back(CCDwidth);
  spar.push_back(CCDwidth);
  //----- Upper CCD 
 IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, &colup, ccdsize*(ccds[0].pt()+0.5*CCDlength*ccds[0].vec()));
  //----- Lower CCD (leftmost point & direction dowel2-dowel1)
 IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, &coldown, ccdsize*(ccds[1].pt()+0.5*CCDlength*ccds[1].vec()) );
  //----- left CCD (uppermost point & direction perpendicular to dowel2-dowel1)
  std::vector<ALIdouble> spar2;
  spar2.push_back(CCDwidth);
  spar2.push_back(CCDdim);
  spar2.push_back(CCDwidth);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar2, &colleft, ccdsize*(ccds[2].pt()+0.5*CCDlength*ccds[2].vec()));
  //----- right CCD (uppermost point & direction perpendicular to dowel2-dowel1)
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar2, &colright, ccdsize*(ccds[3].pt()+0.5*CCDlength*ccds[3].vec()));

}
#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCOPS::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*1.*cm/m ); //COCOA internal units are meters
}
