//   COCOA class implementation file
//Id:  OptOSensor2D.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOSensor2D.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h" 
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaModel/interface/DeviationsFromFileSensor2D.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>		// include floating-point std::abs functions

using namespace CLHEP;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as distance to previous object 'screen'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::defaultBehaviour( LightRay& lightray, Measurement& meas )
{
  makeMeasurement( lightray, meas);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as distance to previous object 'screen'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::makeMeasurement( LightRay& lightray, Measurement& meas ) 
{
  //---------- check is last and not the only one
  /*    if( vocite - OptOList().begin() == 0 ||
    OptOList().end() - vocite != 1) {
    std::cerr << " last and not only one Optical Object should be 'sensor2D' (unless you specify (:T)raverse) " <<
    OptOList().end() - vocite << " size " <<OptOList().size() << std::endl;
    DumpBadOrderOptOs(); 
    }*/

  //---------- Get simulated value 
  //---------- Get intersection
  CLHEP::Hep3Vector ZAxis(0.,0,1.);
  CLHEP::HepRotation rmt = rmGlob();
  ZAxis = rmt * ZAxis;
  if( ALIUtils::debug >= 4 ) {
    ALIUtils::dump3v( centreGlob(), " sensor2D centre Glob ");
    ALIUtils::dump3v( ZAxis, " snsor2D normal ");
  }
  //-  ALIUtils::dumprm( rmt, "rot global " );
  lightray.intersect( ALIPlane(centreGlob(), ZAxis) );
  CLHEP::Hep3Vector inters = lightray.point();
  
  ALIdouble* interslc;
  interslc = convertPointToLocalCoordinates( inters );
  ALIdouble interslcx = interslc[0];
  ALIdouble interslcy = interslc[1];
  meas.setValueSimulated( 0, interslcx );
  meas.setValueSimulated( 1, interslcy );
  
  //----- Dump info
  if (ALIUtils::debug >= 2) {
    //--- Special for range studies
    ALIstring chrg = "";
    /*t    if(Model::Ncmslinkrange >= 1 && Model::Ncmslinkrange <= 8 ) {
      chrg = "RG";
    } else {
      chrg = "";
      } t*/
    CLHEP::Hep3Vector measvv( meas.value()[0], meas.value()[1], 0.);
    measvv = rmt*measvv;
    ALIUtils::dump3v( measvv, " $$$$$$MEAS IN LOCAL FRAME");
    ALIUtils::dump3v( measvv+centreGlob(), " $$$$$$MEAS IN GLOBAL FRAME");

    ALIdouble detH =  1000*meas.valueSimulated(0); if(std::abs(detH) <= 1.e-9 ) detH = 0.;
    ALIdouble detV =  1000*meas.valueSimulated(1); if(std::abs(detV) <= 1.e-9 ) detV = 0.;
    std::cout << "REAL value: " << chrg << meas.valueType(0) << ": " << 1000*meas.value()[0] << chrg << " " << meas.valueType(1) << ": " << 1000*meas.value()[1]  << " (mm)  " << (this)->name() 
	 << "   DIFF= " << detH-1000*meas.value()[0] << " " << detV-1000*meas.value()[1] << std::endl;
    std::cout << "SIMU value: " << chrg << " " << meas.valueType(0) << ": "
      // << setprecision(3) << setw(4)
	      << detH 
	      << chrg << " " << meas.valueType(1) << ": " << detV 
	      << " (mm)  " << (this)->name() << std::endl;
    /*-    std::cout << "SIMU value: " << chrg << " " << meas.valueType(0) << ": "
      // << setprecision(3) << setw(4)
      	      << detH / 0.3125
	      << chrg << " " << meas.valueType(1) << ": " << detV / 0.3125
	      << " STRIPS  " << (this)->name() << std::endl; */
    //	 << detH 
    //	 << chrg << " V: " << detV 
    //	 << " (mm)  " << (this)->name() << std::endl;
    ALIUtils::dump3v( 1000.*(inters - parent()->centreGlob()) , " $$$$$$SIMU inters - parent centre");
    ALIUtils::dump3v(  1000.*(inters - centreGlob()) , " $$$$$$SIMU inters - centre");
  }
  //t    delete &lightray;    
  
  // store the lightray position and direction
  meas.setLightRayPosition( lightray.point() );
  meas.setLightRayDirection( lightray.direction() );

  delete[] interslc;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of Light Ray traverses
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::fastTraversesLightRay( LightRay& lightray )
{
  verbose = ALIUtils::debug;
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST TRAVERSES SENSOR2D  " << name() << std::endl;

  //---------- Shift and Deviate

  //---------- Get intersection 
  CLHEP::Hep3Vector ZAxis(0.,0,1.);
  CLHEP::HepRotation rmt = rmGlob();
  ZAxis = rmt * ZAxis;
  lightray.intersect( ALIPlane(centreGlob(), ZAxis) );
  CLHEP::Hep3Vector inters = lightray.point();

  ALIdouble deviX, deviY, devi;
  // if deviationsFromFile are applied and this sensors has them get the deviations that corresponds to the intersection point.
  // Set this deviation as original one, as it will not be changed for derivatives
  if( DeviationsFromFileSensor2D::apply() && fdevi_from_file) {
    //-    std::cout << " DeviationsFromFileSensor2D::apply() " << DeviationsFromFileSensor2D::apply() << std::endl;
    if(ALIUtils::debug >= 4) {
      std::cout << "fdeviFromFile" << fdevi_from_file << std::endl;
      //-      std::cout << "deviFromFile" << deviFromFile << std::endl;
    }
    //--------- get measurement value of the current sensor
    std::vector< Measurement* >& measv = Model::MeasurementList();
    unsigned int ii;
    Measurement *omeas = 0;
    for( ii = 0; ii < measv.size(); ii++ ) {
      //-   std::cout << " sensor2d finding meas " <<  measv[ii]->sensorName() << " " << name() << std::endl;
      if( measv[ii]->sensorName() == name() ) {
	omeas = measv[ii];
	break;
      }
    }
    if( omeas == 0 ) {
      std::cerr << "!!!EXITING OptOSensor2D::fastTraversesLightRay: meas " << name() << " not found " << std::endl;
    }

    ALIdouble interslcx = omeas->value( 0 );
    ALIdouble interslcy = omeas->value( 1 );
    if(ALIUtils::debug >= 5) std::cout << " interslcx " << interslcx << " interslcy " << interslcy << std::endl;
    //----- transform in milimeters and positive
    //mum    interslcx = interslcx*1.E6 + 10000.;
    //mum    interslcy = interslcy*1.E6 + 10000.;
    ALIdouble df = ALIUtils::LengthValueDimensionFactor();
    interslcx = interslcx/df + 0.010/df;
    interslcy = interslcy/df + 0.010/df;
     if(ALIUtils::debug >= 5) std::cout << " interslcx " << interslcx << " interslcy " << interslcy << std::endl;

    //---------- Get deviations from file (they are in microrads)
    std::pair< ALIdouble, ALIdouble> devis = deviFromFile->getDevis( interslcx, interslcy );
    deviX = devis.second;
    deviY = -devis.first;
     // deviX = devis.first;
     //  deviY = devis.second;
    
    //o  deviX = *((deviFromFile->deviationsInX()).begin() + pointY*deviFromFile->nPoints() + pointX) * 1.E-6;;
    //o  deviY = *((deviFromFile->deviationsInY().begin()) + pointY*deviFromFile->nPoints() + pointX) * 1.E-6;
    
    //---------- Set this deviation value as original one, as it will not be changed for derivatives (change the entry and also the ExtraEntryValueOriginalList())
    ALIuint entryNo = extraEntryNo( "deviX" );
    if( verbose >= 3 ) std::cout << "entrynox" << entryNo << name() << verbose << std::endl;
    Entry* entryDeviX = *(ExtraEntryList().begin()+entryNo);
    entryDeviX->setValue( deviX );
    //-    std::vector< ALIdouble >::const_iterator eevolite = static_cast<std::vector< ALIdouble >::iterator>( ExtraEntryValueOriginalList().begin() );
    std::vector< ALIdouble > eevil = ExtraEntryValueOriginalList();
    //-    std::vector< ALIdouble >::const_iterator eevolite = ( ExtraEntryValueOriginalList().begin() );
    std::vector< ALIdouble >::iterator eevolite = eevil.begin();

    *(eevolite+entryNo) = deviX;
    if( verbose >= 3 ) std::cout<< " entryDeviX name " << entryDeviX->name() << entryDeviX->value() << std::endl;
    entryNo = extraEntryNo( "deviY" );
    Entry* entryDeviY = *(ExtraEntryList().begin()+entryNo);
    //- std::cout << "entrynoy" << entryNo << name() << std::endl;
    entryDeviY->setValue( deviY );
    *(eevolite+entryNo) = deviY;
    //-  std::cout<< entryDeviY << " entryDeviY name " << entryDeviY->name() << entryDeviY->value() << std::endl;
    
  } else {
    deviX = findExtraEntryValue("deviX");
    deviY = findExtraEntryValue("deviY");

    //??? why previous does not work??
    if( fdevi_from_file ) {
      if( ALIUtils::debug >= 5) std::cout << "fdeviFromFile" << fdevi_from_file << std::endl;
      ALIuint entryNo = extraEntryNo( "deviX" );
      Entry* entryDeviX = *(ExtraEntryList().begin()+entryNo);
     if( verbose >= 3 ) std::cout<< entryDeviX << " entryDeviX name " << entryDeviX->name() << entryDeviX->value() << std::endl;
      deviX = entryDeviX->value();
      entryNo = extraEntryNo( "deviY" );
      Entry* entryDeviY = *(ExtraEntryList().begin()+entryNo);
      if( verbose >= 3 )  std::cout<< entryDeviY << " entryDeviY name " << entryDeviY->name() << entryDeviY->value() << std::endl;
      deviY = entryDeviY->value();

    } else {
      ALIbool bb = findExtraEntryValueIfExists("devi", devi);
      if( bb ) {
	deviX = devi;
	deviY = devi;
      }
    }
  }
  if(ALIUtils::debug >= 4) {
    std::cout << "devi " << devi << " devi x  " << deviX << " devi y  " << deviY << std::endl;
  }

  lightray.setPoint( inters );

  lightray.shiftAndDeviateWhileTraversing( this, 'T' );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Shifted and Deviated");
  }

}      


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of Light Ray traverses
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::detailedTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 4) std::cout << "%%% LR: DETAILED TRAVERSES SENSOR2D  " << name() << std::endl;
  if( DeviationsFromFileSensor2D::apply() && fdevi_from_file) {
   DeviationsFromFileSensor2D::setApply( 0 );
   //- std::cout << "fdeviFromFile" << fdevi_from_file << std::endl;
    if(ALIUtils::debug >= 0 )std::cerr << "!!WARNING: sensor " << name() << " has read deviation from file and it will not be taken into account. Please use FAST TRAVERSES" << deviFromFile << std::endl;
  }

  //---------- If width is 0, just keep the intersection point 
  ALIdouble width = findExtraEntryValue("width");
  if( width == 0 ) {
    //---------- Get intersection 
    CLHEP::Hep3Vector ZAxis(0.,0,1.);
    CLHEP::HepRotation rmt = rmGlob();
    ZAxis = rmt * ZAxis;
    lightray.intersect( ALIPlane(centreGlob(), ZAxis) );
    CLHEP::Hep3Vector inters = lightray.point();
    lightray.setPoint( inters );
    if (ALIUtils::debug >= 2) {
      lightray.dumpData("LightRay Sensor2D traversed: "); 
    }
    return;
  }

  if (ALIUtils::debug >= 4) std::cout << std::endl << "$$$ LR: REFRACTION IN FORWARD PLATE " << std::endl;
  //---------- Get forward plate
  ALIPlane plate = getPlate(1, 1);
  //---------- Refract while entering object
  ALIdouble refra_ind1 = 1.;
  ALIdouble refra_ind2 = findExtraEntryValueMustExist("refra_ind");
  lightray.refract( plate, refra_ind1, refra_ind2 );

  if (ALIUtils::debug >= 4) std::cout << std::endl << "$$$ LR: REFRACTION IN BACKWARD PLATE " << std::endl;
  //---------- Get backward plate
  plate = getPlate(0, 1);
  //---------- Refract while exiting splitter
  lightray.refract( plate, refra_ind2, refra_ind1 );

  CLHEP::Hep3Vector inters = lightray.point();
  lightray.setPoint( inters );

  if (ALIUtils::debug >= 4) {
    lightray.dumpData("LightRay Sensor2D traversed: "); 
  }


}     


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ fillExtraEntry: fill it from file or fill it the usual way
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::fillExtraEntry( std::vector<ALIstring>& wordlist )
{

  if(ALIUtils::debug >= 5) std::cout << "OptOSensor2D fillExtraEntry wordlist[1] " << wordlist[1] << std::endl;
  //---------- check if it is deviation read from file 
  fdevi_from_file = 0;
  //-  std::cout << "WL " << wordlist[1]<< "WL " << wordlist[2]<< "WL " << wordlist[3] << std::endl;
  if( wordlist[1] == ALIstring("devi") && wordlist[2] == ALIstring("from_file") ) {
    //---------- Open file
    ALIstring fnam;
    if( wordlist.size() >= 4) {
      fnam = wordlist[3];
    } else {
    //----- Build up file name if it does not exists
      fnam = "dat/devi-";
      fnam += shortName(); 
      fnam += ".dat";
    }

    ALIFileIn& ifdevi = ALIFileIn::getInstance( fnam );

    //----- Check that file exists 
    if(ALIUtils::debug >= 4) std::cout << "Opening deviation file: " << fnam << std::endl;
    /*-    if( !ifdevi ) {
      std::cerr << " !!! Sensor2D Deviation file not found: " << fnam << " of object " << name() << std::endl;
      exit(1);
      }*/

    deviFromFile = new DeviationsFromFileSensor2D();
    fdevi_from_file = 1;
    if(ALIUtils::debug >= 5 ) std::cout << "deviFromFile " << deviFromFile << std::endl; 
    //----- Read header
    ALIstring sensor1_name, sensor2_name;
    ALIdouble sensor_dist;
    ALIdouble prec_deviX,prec_deviY;

    std::vector<ALIstring> wl;
    ifdevi.getWordsInLine( wl );
    sensor1_name = wl[0];
    sensor2_name = wl[1];
    sensor_dist = atof( wl[2].c_str() );
    // 'c' means that light is traversing the glass
    if(sensor1_name[sensor1_name.size()-2] == 'c') {
      sensor1_name = sensor1_name.substr(0,sensor1_name.size()-1);
    }
    if(sensor2_name[sensor2_name.size()-2] == 'c') {
      sensor2_name = sensor2_name.substr(0,sensor2_name.size()-1);
    }
    if(ALIUtils::debug >= 5) std::cout << "sensor1_name " << sensor1_name << " sensor2_name " << sensor2_name  << " sensor_dist " << sensor_dist << " unknown " << wl[3] << std::endl;

    ifdevi.getWordsInLine( wl );
    prec_deviX = atof( wl[0].c_str() );
    prec_deviY = atof( wl[1].c_str() );

    if(ALIUtils::debug >= 5) std::cout << "prec_deviX " <<  prec_deviX  << " prec_deviY " << prec_deviY << std::endl;

    deviFromFile = new DeviationsFromFileSensor2D();
    ALIdouble offsetX, offsetY;
    if( wl.size() == 5 ) {
      offsetX = ALIUtils::getFloat( wl[3] );
      offsetY = ALIUtils::getFloat( wl[4] );
      deviFromFile->setOffset( offsetX, offsetY );
    }
    deviFromFile->readFile( ifdevi );
    fdevi_from_file = 1;
    if(ALIUtils::debug >= 5 ) std::cout << "deviFromFile " << deviFromFile << std::endl; 

 
    //--- Fill extra entries 'deviX' & 'deviY' to compute derivatives 
    std::vector<ALIstring> wlo; 
    char chartmp[20];
    wlo.push_back( wordlist[0] );
    wlo.push_back("deviX");
    wlo.push_back("0"); // value is set to 0 as it is on the file and the point of intersection is not computed yet
    gcvt( prec_deviX, 10, chartmp );
    wlo.push_back( ALIstring(chartmp) );
    wlo.push_back("cal");
    std::vector<ALIstring> wl2(wlo); 
    OpticalObject::fillExtraEntry( wlo );

    wl2[1] = "deviY";   
    gcvt( prec_deviY, 10, chartmp );
    wl2[3] = ALIstring( chartmp );
    OpticalObject::fillExtraEntry( wl2 ); 

  } else {
    OpticalObject::fillExtraEntry( wordlist );
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble* OptOSensor2D::convertPointToLocalCoordinates( const CLHEP::Hep3Vector& point)
{
  ALIdouble* interslc = new ALIdouble[2];
  
  //----- X value
  CLHEP::HepRotation rmt = rmGlob();
  CLHEP::Hep3Vector XAxism(1.,0.,0.);
  XAxism*=rmt;
  if( ALIUtils::debug >= 5) ALIUtils::dump3v( (this)->centreGlob(),  "centre glob sensor2D" );
  if( ALIUtils::debug >= 5) ALIUtils::dumprm( rmt,  "rotation matrix sensor2D" );
  //t  ALIUtils::dump3v(point - (this)->centreGlob() , "inters - (this)->centreGlob()");
  if( ALIUtils::debug >= 5) ALIUtils::dump3v(XAxism , "XAxism");
  interslc[0] = (point - (this)->centreGlob() ) * XAxism;
  
  //----- Y value
  CLHEP::Hep3Vector YAxism(0.,1.,0.);
  YAxism*=rmt;
  if( ALIUtils::debug >= 5)
ALIUtils::dump3v(YAxism , "YAxism");
  interslc[1] = (point - (this)->centreGlob() ) * YAxism;
  
  if( ALIUtils::debug >= 5 ) {
    std::cout << " intersection in local coordinates: X= " << interslc[0] << "  Y= " << interslc[1] << std::endl;
    ALIUtils::dump3v( point - (this)->centreGlob() , " inters - centre " );
  }
  return interslc;
}

#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::fillVRML()
{
  //-  std::cout << " filling optosensor " << std::endl;
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour( 0., 0., 1., 0. );
  vrmlmgr.AddBox( *this, 1., 1., .2, col);
  vrmlmgr.SendReferenceFrame( *this, 0.6); 
  vrmlmgr.SendName( *this, 0.1 );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 0., 1., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(20.);
  spar.push_back(20.);
  spar.push_back(5.);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
}
#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSensor2D::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*4.*cm/m, go*4.*cm/m, go*1.*cm/m ); //COCOA internal units are meters
}
