//   COCOA class implementation file
//Id:  OpticalObject.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/OpticalObjectMgr.h"
#include "Alignment/CocoaModel/interface/OptOLaser.h"
#include "Alignment/CocoaModel/interface/OptOSource.h"
#include "Alignment/CocoaModel/interface/OptOXLaser.h"
#include "Alignment/CocoaModel/interface/OptOMirror.h"
#include "Alignment/CocoaModel/interface/OptOPlateSplitter.h"
#include "Alignment/CocoaModel/interface/OptOCubeSplitter.h"
#include "Alignment/CocoaModel/interface/OptOModifiedRhomboidPrism.h"
#include "Alignment/CocoaModel/interface/OptOOpticalSquare.h"
#include "Alignment/CocoaModel/interface/OptOLens.h"
#include "Alignment/CocoaModel/interface/OptORisleyPrism.h"
#include "Alignment/CocoaModel/interface/OptOSensor2D.h"
#include "Alignment/CocoaModel/interface/OptODistancemeter.h"
#include "Alignment/CocoaModel/interface/OptODistancemeter3dim.h"
#include "Alignment/CocoaModel/interface/OptOScreen.h"
#include "Alignment/CocoaModel/interface/OptOTiltmeter.h"
#include "Alignment/CocoaModel/interface/OptOPinhole.h"
#include "Alignment/CocoaModel/interface/OptOCOPS.h"
#include "Alignment/CocoaModel/interface/OptOUserDefined.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"

#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaModel/interface/EntryLengthAffCentre.h"
#include "Alignment/CocoaModel/interface/EntryAngleAffAngles.h"
#include "Alignment/CocoaModel/interface/EntryNoDim.h"
#include "Alignment/CocoaModel/interface/EntryMgr.h"

#include "Alignment/CocoaDDLObjects/interface/CocoaMaterialElementary.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"

#include "CLHEP/Units/SystemOfUnits.h"

#include <stdlib.h>
#include <fstream>
#include <iostream>


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Constructor: set OptO parent, type, name and fcopyData (if data is read or copied)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalObject::OpticalObject( OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data ) :theParent(parent), theType(type), theName(name), fcopyData( copy_data)
{
  if ( ALIUtils::debug >= 4 ) {
    std::cout << std::endl << "@@@@ Creating OpticalObject: NAME= " << theName << " TYPE= " <<theType << " fcopyData " <<fcopyData <<std::endl;
  }

  OpticalObjectMgr::getInstance()->registerMe( this );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ construct: read centre and angles from file (or copy it) and create component OptOs
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::construct()
{
  //---------- Get file handler
  ALIFileIn& filein = ALIFileIn::getInstance( Model::SDFName() );
  /*-  if(!filein) {
    filein.ErrorInLine();
    std::cerr << "cannot open file SystemDescription.txt" << std::endl; 
    exit(0);
    }*/

  if( theParent != 0 ) { //----- OptO 'system' has no parent (and no affine frame)
    //---------- Read or copy Data
    if(!fcopyData) { 
      if(ALIUtils::debug >=4) std::cout << "@@@@ Reading data of Optical Object " << name() << std::endl;
      readData( filein );
    } else {
      if(ALIUtils::debug >=4) std::cout << "Copy data of Optical Object " << name() << std::endl;
      copyData();
    }

    //---------- Set global coordinates 
    setGlobalCoordinates();
    //---------- Set ValueDisplacementByFitting to 0. !!done at Entry construction
    /*    std::vector<Entry*>::const_iterator vecite;
    for ( vecite = CoordinateEntryList().begin(); vecite != CoordinateEntryList().end(); vecite++) {
      (*vecite)->setValueDisplacementByFitting( 0. );
    }
    */

    //---------- Set original entry values
    setOriginalEntryValues();
  }

  //---------- Create the OptO that compose this one
  createComponentOptOs( filein );

  //---------- Construct material
  constructMaterial();

  //---------- Construct solid shape
  constructSolidShape();

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Reads the affine frame (centre and angles) that every OpticalObject should have
//@@ If type is source, it may only read the center (and set the angles to 0)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::readData( ALIFileIn& filein )
{
  //---------- See if there are extra entries and read them
  std::vector<ALIstring> wordlist;
  filein.getWordsInLine( wordlist );
  if (wordlist[0] == ALIstring( "ENTRY" ) ){
    //---------- Read extra entries from file
    readExtraEntries( filein );
    filein.getWordsInLine( wordlist );
  } 

  //--------- set centre and angles not global (default behaviour)
  centreIsGlobal = 0;
  anglesIsGlobal = 0;

  //--------- readCoordinates 
  if ( type() == ALIstring("source") || type() == ALIstring("pinhole") ) {
    readCoordinates( wordlist[0], "centre", filein );
    setAnglesNull();
  } else {
 //---------- Read centre and angles
    readCoordinates( wordlist[0], "centre", filein );
    filein.getWordsInLine( wordlist );
    readCoordinates( wordlist[0], "angles", filein );
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ ReadExtraEntries: Reads extra entries from file
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::readExtraEntries( ALIFileIn& filein ) 
{
  //---------- Loop extra entries until a '}'-line is found
  std::vector<ALIstring> wordlist;
  for (;;) {
    filein.getWordsInLine( wordlist );
    if ( wordlist[0] != ALIstring("}") ) {  
      fillExtraEntry( wordlist );
    } else {
      break;
    }
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ fillExtraEntry: create and fill an extra entry. Put it in lists and set link OptO it belongs to
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::fillExtraEntry( std::vector<ALIstring>& wordlist )
{

  //- if(ALIUtils::debug >= 5) std::cout << "fillExtraEntry wordlist0 " << wordlist[0].c_str() << std::endl;

  //---------- Check which type of entry to create 
  Entry* xentry;
  if ( wordlist[0] == ALIstring("length") ) {
    xentry = new EntryLength( wordlist[0] );
  } else if ( wordlist[0] == ALIstring("angle") ) {
    xentry = new EntryAngle( wordlist[0] );
  } else if ( wordlist[0] == ALIstring("nodim") ) {
    xentry = new EntryNoDim( wordlist[0] );
  } else { 
    std::cerr << "!!ERROR: Exiting...  unknown type of Extra Entry " << wordlist[0] << std::endl;
    ALIUtils::dumpVS( wordlist, " Only 'length', 'angle' or 'nodim' are allowed ", std::cerr );
    exit(2);
  }

  if( ALIUtils::debug>=99) {
    ALIUtils::dumpVS( wordlist, "fillExtraEntry: ", std::cout );
  }
  //---------- Erase first word of line read (type of entry)
  wordlist.erase( wordlist.begin() );

  if( ALIUtils::debug>=99) {
    ALIUtils::dumpVS( wordlist, "fillExtraEntry: ", std::cout );
  }

  //---------- Set link from entry to OptO it belongs to
  xentry->setOptOCurrent( this );
  //----- Name is filled from here to be consistent with fillCoordinate
  xentry->fillName( wordlist[0] );
  //---------- Fill entry with data in line read 
  xentry->fill( wordlist );

  //---------- Add entry to entry lists
  Model::addEntryToList( xentry );
  addExtraEntryToList( xentry );

  if(ALIUtils::debug >=5) std::cout << "fillExtraEntry: xentry_value" <<  xentry->value()<<xentry->ValueDimensionFactor() << std::endl;

  //---------- Add entry value to list 
  addExtraEntryValueToList( xentry->value() );

  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ readCoordinates: Read centre or angles ( following coor_type )
//@@ check that coordinate type is the expected one  
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::readCoordinates( const ALIstring& coor_type_read, const ALIstring& coor_type_expected, ALIFileIn& filein )
{

  ALIstring coor_type_reads = coor_type_read.substr(0,6);
  if( coor_type_reads == "center" ) coor_type_reads = "centre";
  //---------- if after the first six letters ther is a 'G', it means they are global coordinates
  //----- If data is read from a 'report.out', it is always local and this is not needed
  
  //TODO: check that if only one entry of the three is read from 'report.out', the input file does not give global coordinates (it would cause havoc)
  if( EntryMgr::getInstance()->findEntryByLongName( longName(), "" ) == 0 ) {
    if(coor_type_read.size() == 7) { 
      if(coor_type_read[6] == 'G' ) {
	if(ALIUtils::debug >= 5) std::cout << " coordinate global " << coor_type_read << std::endl;
	if(coor_type_expected == "centre" ) {
	  centreIsGlobal = 1;
	} else if(coor_type_expected == "angles" ) {
	  anglesIsGlobal = 1;
	}
      }
    }
  }

  std::vector<ALIstring> wordlist;
  //---------- Read 4 lines: first is entry type, rest are three coordinates (each one will be an individual entry)
  ALIstring coor_names[3]; // to check if using cartesian, cylindrical or spherical coordinates

  for( int ii=0; ii<4; ii++ ) {  
    if ( ii == 0 ) {
      //---------- Check that first line is of expected type
      if ( coor_type_reads != coor_type_expected ) { 
        filein.ErrorInLine();
        std::cerr << "readCoordinates: " << coor_type_expected << " should be read here, instead of " << coor_type_reads << std::endl;
        exit(1);
      }
    } else {
      //----------- Fill entry Data
      filein.getWordsInLine(wordlist);
      coor_names[ii-1] = wordlist[0];
      fillCoordinateEntry( coor_type_expected, wordlist );
    }
  }

  //---- Transform coordinate system if cylindrical or spherical
  if( coor_names[0] == ALIstring("X") && coor_names[1] == ALIstring("Y") && coor_names[2] == ALIstring("Z")) {
    //do nothing
  }else if( coor_names[0] == ALIstring("R") && coor_names[1] == ALIstring("PHI") && coor_names[2] == ALIstring("Z")) {
    transformCylindrical2Cartesian();
  }else if( coor_names[0] == ALIstring("R") && coor_names[1] == ALIstring("THE") && coor_names[2] == ALIstring("PHI")) {
    transformSpherical2Cartesian();
  } else {
    std::cerr << "!!!EXITING: coordinates have to be cartesian (X ,Y ,Z), or cylindrical (R, PHI, Z) or spherical (R, THE, PHI) " << std::endl
	 << " they are " << coor_names[0] << ", " << coor_names[1] << ", " << coor_names[2] << "." << std::endl;
    exit(1);
  }
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void OpticalObject::transformCylindrical2Cartesian()
{
  ALIuint ii;
  ALIuint siz =  theCoordinateEntryVector.size();
  ALIdouble newcoor[3];
  ALIdouble R = theCoordinateEntryVector[0]->value();
  ALIdouble phi = theCoordinateEntryVector[1]->value()/ALIUtils::LengthValueDimensionFactor()*ALIUtils::AngleValueDimensionFactor();
  newcoor[0] = R*cos(phi);
  newcoor[1] = R*sin(phi);
  newcoor[2] = theCoordinateEntryVector[2]->value(); // Z
  //-  std::cout << " phi " << phi << std::endl;
   //----- Name is filled from here to include 'centre' or 'angles'

 for( ii = 0; ii < siz; ii++ ) { 
    if(ALIUtils::debug >= 5 ) std::cout << " OpticalObject::transformCylindrical2Cartesian " << ii << " " << newcoor[ii] << std::endl;
    theCoordinateEntryVector[ii]->setValue( newcoor[ii] );
  }
 // change the names
  ALIstring name = "centre_X";
  theCoordinateEntryVector[0]->fillName( name );
  name = "centre_Y";
  theCoordinateEntryVector[1]->fillName( name );
  name = "centre_Z";
  theCoordinateEntryVector[2]->fillName( name );

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void OpticalObject::transformSpherical2Cartesian()
{

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ fillCoordinateEntry: fill data of Coordinate Entry with data read 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::fillCoordinateEntry( const ALIstring& coor_type, const std::vector<ALIstring>& wordlist )
{
  //---------- Select which type of entry to create
  Entry* entry = 0;
  if ( coor_type == ALIstring("centre") ) {
    entry = new EntryLengthAffCentre( coor_type );
  }else if ( coor_type == ALIstring("angles") ) {
    entry = new EntryAngleAffAngles( coor_type );
  }

  //---------- Set link from entry to OptO it belongs to
  entry->setOptOCurrent( this );
  //----- Name is filled from here to include 'centre' or 'angles'
  ALIstring name = coor_type + "_" + wordlist[0];
  entry->fillName( name );
  //---------- Fill entry with data read
  entry->fill( wordlist );

  //---------- Add entry to lists
  Model::addEntryToList( entry );
  addCoordinateEntryToList( entry );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ SetAnglesNull: create three angle entries and set values to zero
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setAnglesNull()
{
  
  EntryAngleAffAngles* entry;
  //---------- three names will be X, Y and Z
  ALIstring coor("XYZ");
  
  //---------- Fill the three entries
  for (int ii=0; ii<3; ii++) {  
    entry = new EntryAngleAffAngles( "angles" );

    //---------- Set link from entry to OptO it belongs to
    entry->setOptOCurrent( this );
    //----- Name is filled from here to be consistent with fillCoordinate
    ALIstring name = "angles_" + coor.substr(ii, 1);
    entry->fillName( name );
    //---------- Set entry data to zero    
    entry->fillNull( );

       // entry.fillNull( tt );

    //---------- Add entry to lists
    Model::addEntryToList( entry );
    addCoordinateEntryToList( entry );

  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ copyData: look for OptO of the same type as this one and copy its components as the components of present OptO
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::copyData()
{
  centreIsGlobal = 0;
  anglesIsGlobal = 0;
  if(ALIUtils::debug >= 5) std::cout << "entering copyData()" << std::endl;

  //---------- Get copied OptO
  OpticalObject* opto = Model::nextOptOToCopy();

  //---------- build name: for a copied OptO, now name is parent name, add the last part of the copied OptO
  ALIint copy_name_last_slash = opto->name().rfind('/');  
  ALIint copy_name_size = opto->name().length();
  //-  if(ALIUtils::debug >= 9) std::cout << "BUILD UP NAME0 " << theName << std::endl;
  theName.append( opto->name(), copy_name_last_slash, copy_name_size);
  if(ALIUtils::debug >= 5) std::cout << "copying OptO: " << opto->name() << " to OptO " << theName << std::endl;

  //---------- Copy Extra Entries from copied OptO
  std::vector<Entry*>::const_iterator vecite;
  for( vecite = opto->ExtraEntryList().begin(); vecite != opto->ExtraEntryList().end(); vecite++ ) {
    std::vector<ALIstring> wordlist;
    wordlist.push_back( (*vecite)->type() );
    buildWordList( (*vecite), wordlist );      
    if( ALIUtils::debug>=9) {
      ALIUtils::dumpVS( wordlist, "copyData: ", std::cout );
    }
    fillExtraEntry( wordlist );
  }

  //---------- Copy Coordinate Entries from copied OptO
  for( vecite = opto->CoordinateEntryList().begin(); vecite != opto->CoordinateEntryList().end(); vecite++ ) {
    std::vector<ALIstring> wordlist;
    buildWordList( (*vecite), wordlist );
    //----- first three coordinates centre, second three coordinates angles!!PROTECT AGAINST OTHER POSSIBILITIES!!
    ALIstring coor_name; 
    if( vecite - opto->CoordinateEntryList().begin() < 3 ) {
      coor_name = "centre";
    } else {
      coor_name = "angles";
    }
    fillCoordinateEntry( coor_name, wordlist );
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ buildWordList:  for copied OptOs, make a list of words to be passed to fillExtraEntry or fill CoordinateEntry
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::buildWordList( const Entry* entry, std::vector<ALIstring>& wordlist )
{
  //---------- 1st add name
  wordlist.push_back( entry->name() );
  
  //---------- 1st add value
  char chartmp[20];
  gcvt(entry->value()/entry->ValueDimensionFactor(),10, chartmp);
  wordlist.push_back( chartmp );

  //---------- 1st add sigma
  gcvt(entry->sigma()/entry->SigmaDimensionFactor(),10, chartmp);
  wordlist.push_back( chartmp );

  //---------- 1st add quality
  ALIstring strtmp;
  ALIint inttmp = entry->quality();
  switch ( inttmp ) {
  case 0:
    strtmp = "fix";
    break;
  case 1:
    strtmp = "cal";
    break;
  case 2:
    strtmp = "unk";
    break;
  default:
    std::cerr << "buildWordList: entry " << entry->OptOCurrent()->name() << entry->name() << " quality not found " << inttmp << std::endl;
    break;
  }
  wordlist.push_back( strtmp );

  if( ALIUtils::debug>=9) {
    ALIUtils::dumpVS( wordlist, "buildWordList: ", std::cout );
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ createComponentOptOs: create components objects of this Optical Object (and call their construct(), so that they read their own data)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::createComponentOptOs( ALIFileIn& filein )
{
  //---------- flag to determine if components are copied or read (it is passed to the constructor of component OptOs)
  ALIbool fcopyComponents = 0; 

  //---------- Get list of components of current OptO (copy it to 'vopto_types') 
  std::vector<ALIstring> vopto_types;
  int igetood = Model::getComponentOptOTypes( type(), vopto_types );
  if( !igetood ) {
    if(ALIUtils::debug >= 5) std::cout << " NO MORE COMPONENTS IN THIS OptO" << name() << std::endl ; 
    return;
  }

  /*  //---------- Dump component list
  if(ALIUtils::debug >= 5) {
    ALIUtils::dumpVS( wordlist, "SYSTEM: ", std::cout );
    }*/
 
  //---------- Loop components (have to follow list in 'vopto_types')
  std::vector<ALIstring>::iterator vsite;
  std::vector<ALIstring> wordlist; 
  for( vsite=vopto_types.begin(); vsite!=vopto_types.end(); vsite++ ) {
    //----- If it is not being copied, read first line describing object 
    //- std::cout << "fcopyy" << fcopyComponents << fcopyData << theName << *vsite << std::endl;
    if( !fcopyData && !fcopyComponents ) filein.getWordsInLine(wordlist);
    //t    if( !fcopyData ) filein.getWordsInLine(wordlist);

    //----- Check first line describing object 
    //--- Don't check it if OptO is going to be copied (fcopyData = 1)
    //--- If OptO is not copied, but components will be copied, check if only for the first component (for the second fcopyComponents=1)
    if( fcopyData || fcopyComponents ) {
      fcopyComponents = 1;
    //--- If OptO not copied, but components will be copied
    }else if( wordlist[0] == ALIstring("copy_components") ) {
      if(ALIUtils::debug>=3)std::cout << "createComponentOptOs: copy_components" << wordlist[0] << std::endl;
      Model::createCopyComponentList( type() );
      fcopyComponents = 1;  //--- for the second and following components
    //----- If no copying: check that type is the expected one
    } else if ( wordlist[0] != (*vsite) ) {
        filein.ErrorInLine();
        std::cerr << "!!! Badly placed OpticalObject: " << wordlist[0] << " should be = " << (*vsite) << std::endl; 
        exit(2);
    }

    //---------- Make composite component name 
    ALIstring component_name = name();
    //----- if component is not going to be copied add name of component to the parent (if OptO is not copied and components will be, wordlist = 'copy-components', there is no wordlist[1]
    if( !fcopyComponents ) {
      component_name += '/';
      component_name += wordlist[1];
    } 
    //    if ( ALIUtils::debug >= 6 ) std::cout << "MAKE NAME " << name() << " TO " << component_name << std::endl;

    //---------- Create OpticalObject of the corresponding type
    OpticalObject* OptOcomponent;
    if( *vsite == "laser" ) {
      OptOcomponent =  
	new OptOLaser( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "source" ) {
      OptOcomponent =  
	new OptOSource( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "Xlaser" ) {
      OptOcomponent =  
	new OptOXLaser( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "mirror" ){
      OptOcomponent =  
	new OptOMirror( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "plate_splitter" ) {
      OptOcomponent =  
	new OptOPlateSplitter( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "cube_splitter" ) {
      OptOcomponent =  
	new OptOCubeSplitter( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "modified_rhomboid_prism" ) {
      OptOcomponent =  
	new OptOModifiedRhomboidPrism( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "pseudo_pentaprism" || *vsite == "optical_square" ) {
      OptOcomponent =  
	new OptOOpticalSquare( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "lens" ) {
      OptOcomponent =  
	new OptOLens( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "Risley_prism" ) {
      OptOcomponent =  
	new OptORisleyPrism( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "sensor2D" ) {
      OptOcomponent =  
	new OptOSensor2D( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "distancemeter" || *vsite == "distancemeter1dim" ) {
      OptOcomponent =  
	new OptODistancemeter( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "distancemeter3dim" ) {
      OptOcomponent =  
	new OptODistancemeter3dim( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "distance_target" ) {
      OptOcomponent =  
	new OptOScreen( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "tiltmeter" ) {
      OptOcomponent =  
	new OptOTiltmeter( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "pinhole" ) {
      OptOcomponent =  
	new OptOPinhole( this, *vsite, component_name, fcopyComponents );
    } else if( *vsite == "COPS" ) {
      OptOcomponent =  
	new OptOCOPS( this, *vsite, component_name, fcopyComponents );
    } else {
      OptOcomponent =  
	//o	new OpticalObject( this, *vsite, component_name, fcopyComponents );
	new OptOUserDefined( this, *vsite, component_name, fcopyComponents );
    }
    //---------- Construct it (read data and 
    OptOcomponent->construct(); 

    //---------- Fill OptO tree and OptO list 
    Model::OptOList().push_back( OptOcomponent ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ SetGlobalCoordinates
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setGlobalCoordinates()
{
  setGlobalCentre();
  setGlobalRM();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setGlobalCentre()
{
  SetCentreLocalFromEntryValues();
  if ( type() != ALIstring("system") && !centreIsGlobal ) {
    SetCentreGlobFromCentreLocal();
  }
  if( centreIsGlobal ){
    std::cerr << "!!!FATAL ERROR: angles in global coordinates not supported momentarily " << std::endl;
    abort();
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setGlobalRM()
{
  SetRMLocalFromEntryValues();
  if( !anglesIsGlobal ) {
    SetRMGlobFromRMLocal();
  }
    
  // Calculate local rot axis with new rm glob
  calculateLocalRotationAxisInGlobal();
  
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setGlobalRMOriginalOriginal(const HepRotation& rmorioriLocal )
{
  HepRotation rmorioriold = rmGlobOriginalOriginal();
  if ( ALIUtils::debug >= 5 ) {
    std::cout << " setGlobalRMOriginalOriginal OptO " << name() << std::endl;
    ALIUtils::dumprm(rmorioriLocal," setGlobalRMOriginalOriginal new local");
    ALIUtils::dumprm(rmGlobOriginalOriginal()," setGlobalRMOriginalOriginal old ");
  }

  SetRMGlobFromRMLocalOriginalOriginal( rmorioriLocal );

  /*  //---- multiplyt it by parent rmGlobOriginalOriginal
  if( parent()->type() != ALIstring("system") ) {
    theRmGlobOriginalOriginal = parent()->rmGlobOriginalOriginal() * theRmGlobOriginalOriginal;
    }*/

  if ( ALIUtils::debug >= 5 ) {
    ALIUtils::dumprm( parent()->rmGlobOriginalOriginal()," parent rmoriori glob  ");
    ALIUtils::dumprm(rmGlobOriginalOriginal()," setGlobalRMOriginalOriginal new ");
  }

  //----------- Reset RMGlobOriginalOriginal() of every component
  std::vector<OpticalObject*> vopto;
  ALIbool igetood = Model::getComponentOptOs(name(), vopto); 
  if( !igetood ) {
    //    std::cout << " NO MORE COMPONENTS IN THIS OptO" << name() << std::endl ; 
    return;
  }
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    HepRotation rmorioriLocalChild = (*vocite)->buildRmFromEntryValuesOriginalOriginal();
    (*vocite)->setGlobalRMOriginalOriginal( rmorioriLocalChild );
    //    (*vocite)->propagateGlobalRMOriginalOriginalChangeToChildren( rmorioriold, rmGlobOriginalOriginal() );
  }
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::propagateGlobalRMOriginalOriginalChangeToChildren( const HepRotation& rmorioriold, const HepRotation& rmoriorinew )
{
  std::cout << " propagateGlobalRMOriginalOriginalChangeToChildren OptO " << name() << std::endl;
  ALIUtils::dumprm(rmGlobOriginalOriginal()," setGlobalRMOriginalOriginal old ");
  theRmGlobOriginalOriginal = rmoriorinew.inverse() * theRmGlobOriginalOriginal;
  theRmGlobOriginalOriginal = rmorioriold * theRmGlobOriginalOriginal;
  ALIUtils::dumprm(rmGlobOriginalOriginal()," setGlobalRMOriginalOriginal new ");
 
 //----------- Reset RMGlobOriginalOriginal() of every component
  std::vector<OpticalObject*> vopto;
  ALIbool igetood = Model::getComponentOptOs(name(), vopto); 
  if( !igetood ) {
    //    std::cout << " NO MORE COMPONENTS IN THIS OptO" << name() << std::endl ; 
    return;
  }
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    //    HepRotation rmoriorid = buildRmFromEntryValues();
    (*vocite)->propagateGlobalRMOriginalOriginalChangeToChildren( rmorioriold, rmoriorinew );
  }  

} 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
HepRotation OpticalObject::buildRmFromEntryValuesOriginalOriginal()
{
  HepRotation rm;
  const OpticalObject* opto_par = this;
  //  if(Model::GlobalOptions()["rotateAroundLocal"] == 0) {
  if(ALIUtils::debug >= 55) std::cout << "rotate with parent: before X " << opto_par->parent()->name() <<" " <<  parent()->getEntryRMangle(XCoor) <<std::endl;
  const std::vector< Entry* >& cel = CoordinateEntryList();
  rm.rotateX( cel[3]->valueOriginalOriginal() );
  if(ALIUtils::debug >= 55) std::cout << "rotate with parent: before Y " << opto_par->parent()->name() <<" " <<  parent()->getEntryRMangle(YCoor) <<std::endl;
  rm.rotateY( cel[4]->valueOriginalOriginal() );
  if(ALIUtils::debug >= 55) std::cout << "rotate with parent: before Z " << opto_par->parent()->name() <<" " <<  parent()->getEntryRMangle(ZCoor) <<std::endl;
  rm.rotateZ( cel[5]->valueOriginalOriginal() );
  //-  rm.rotateZ( getEntryRMangle(ZCoor) );
  if(ALIUtils::debug >= 54) ALIUtils::dumprm( theRmGlob, ("SetRMGlobFromRMLocal: RM GLOB after " +  opto_par->parent()->longName()).c_str() );

  return rm;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::SetCentreLocalFromEntryValues()
{

  //  std::vector<Entry*>::const_iterator vecite = CoordinateEntryList().begin(); 
  //-  std::cout << "PARENTSYSTEM" << name() << parent() <<"ZZ"<<vecite<< std::endl;
  //  std::cout << " OpticalObject::setGlobalCoordinates " << this->name() << std::endl;
      //-      std::cout << veite << "WW" << *vecite << std::endl;
//---------------------------------------- Set global centre
//----------------------------------- Get local centre from Entries
  theCentreGlob.setX( getEntryCentre(XCoor) );
  theCentreGlob.setY( getEntryCentre(YCoor) );
  theCentreGlob.setZ( getEntryCentre(ZCoor) );
  if(ALIUtils::debug >=4) ALIUtils::dump3v( centreGlob(), "SetCentreLocalFromEntryValues: CENTRE LOCAL ");
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::SetRMLocalFromEntryValues()
{

  //---------- Set global rotation matrix
  //-------- Get rm from Entries
  theRmGlob = CLHEP::HepRotation();
  theRmGlob.rotateX( getEntryRMangle(XCoor) );
  if(ALIUtils::debug >= 4) {
    std::cout << "  getEntryRMangle(XCoor) )" << getEntryRMangle(XCoor) << std::endl;
    ALIUtils::dumprm( theRmGlob, "SetRMLocalFromEntryValues: RM after X");
  }
  theRmGlob.rotateY( getEntryRMangle(YCoor) );
  if(ALIUtils::debug >= 4) {
    std::cout << "  getEntryRMangle(YCoor) )" << getEntryRMangle(YCoor) << std::endl;
    ALIUtils::dumprm( theRmGlob, "SetRMLocalFromEntryValues: RM after Y");
  }
  theRmGlob.rotateZ( getEntryRMangle(ZCoor) );
  if(ALIUtils::debug >= 4) {
    std::cout << "  getEntryRMangle(ZCoor) )" << getEntryRMangle(ZCoor) << std::endl;
    ALIUtils::dumprm( theRmGlob, "SetRMLocalFromEntryValues: RM FINAL");
  }

  //----- angles are relative to parent, so rotate parent angles first 
  //  RmGlob() = 0;
  //-  if(ALIUtils::debug >= 4) ALIUtils::dumprm( parent()->rmGlob(), "OPTO0: RM LOCAL ");
  //  if ( type() != ALIstring("system") ) theRmGlob.transform( parent()->rmGlob() );
  //----- if anglesIsGlobal, RM is already in global coordinates, else multiply by ancestors

  /*  /////////////
  CLHEP::Hep3Vector ztest(0.,0.,1.);
  ztest = theRmGlob * ztest;
  if( ALIUtils::debug >= 5 ) ALIUtils::dump3v( ztest, "z rotated by theRmGlob ");
  */
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::SetCentreGlobFromCentreLocal()
{
  //----------- Get global centre: parent centre plus local centre traslated to parent coordinate system
  CLHEP::Hep3Vector cLocal = theCentreGlob;
  theCentreGlob = parent()->rmGlob() * theCentreGlob;
  
  if(ALIUtils::debug >= 5) ALIUtils::dump3v( theCentreGlob, "SetCentreGlobFromCentreLocal: CENTRE in parent local frame ");
  theCentreGlob += parent()->centreGlob();

  if(ALIUtils::debug >= 5) ALIUtils::dump3v( theCentreGlob, "SetCentreGlobFromCentreLocal: CENTRE GLOBAL ");
  if(ALIUtils::debug >= 5) {
    ALIUtils::dump3v( parent()->centreGlob(), ( " parent centreGlob" + parent()->name() ).c_str() );
    ALIUtils::dumprm( parent()->rmGlob(), " parent rmGlob ");
  }
  CLHEP::Hep3Vector cLocal2 = theCentreGlob - parent()->centreGlob();
  CLHEP::HepRotation rmParentInv = inverseOf( parent()->rmGlob() );
  cLocal2 = rmParentInv * cLocal2;
  if( (cLocal2 - cLocal).mag() > 1.E-9 ) {
    std::cerr << "!!!! CALCULATE LOCAL WRONG. Diff= " << (cLocal2 - cLocal).mag() << " " << cLocal2 << " " << cLocal << std::endl;
    exit(1);
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::SetRMGlobFromRMLocal()
{
  const OpticalObject* opto_par = this;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if(gomgr->GlobalOptions()["rotateAroundLocal"] == 0) {
    while ( opto_par->parent()->type() != ALIstring("system") ) {
      //t    vecite = opto_par->parent()->GetCoordinateEntry( CEanglesX ); 
      if(ALIUtils::debug >= 5) std::cout << "rotate with parent: before X " << opto_par->parent()->name() <<" " <<  parent()->getEntryRMangle(XCoor) <<std::endl;
      theRmGlob.rotateX( parent()->getEntryRMangle(XCoor) );
      if(ALIUtils::debug >= 5) std::cout << "rotate with parent: before Y " << opto_par->parent()->name() <<" " <<  parent()->getEntryRMangle(YCoor) <<std::endl;
      theRmGlob.rotateY( parent()->getEntryRMangle(YCoor) );
      if(ALIUtils::debug >= 5) std::cout << "rotate with parent: before Z " << opto_par->parent()->name() <<" " <<  parent()->getEntryRMangle(ZCoor) <<std::endl;
      theRmGlob.rotateZ( parent()->getEntryRMangle(ZCoor) );
      if(ALIUtils::debug >= 4) ALIUtils::dumprm( theRmGlob, ("SetRMGlobFromRMLocal: RM GLOB after " +  opto_par->parent()->longName()).c_str() );
      opto_par = opto_par->parent();
    }
  }else {
    if(ALIUtils::debug >= 4) {
      std::cout << " Composing rmGlob with parent " <<  parent()->name() << std::endl;
      //      ALIUtils::dumprm( theRmGlob, "SetRMGlobFromRMLocal: RM GLOB ");
    }
    theRmGlob = parent()->rmGlob() * theRmGlob;
  }

    //    std::cout << "rotate with parent (parent)" << opto_par->name() <<parent()->name() << (*vecite)->name() << (*vecite)->value() <<std::endl;
  if(ALIUtils::debug >= 4) {
    ALIUtils::dumprm( theRmGlob, "SetRMGlobFromRMLocal: final RM GLOB ");
    ALIUtils::dumprm(  parent()->rmGlob(), "parent rm glob ");
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::SetRMGlobFromRMLocalOriginalOriginal( HepRotation rmoriori )
{

  theRmGlobOriginalOriginal = rmoriori;
  theRmGlobOriginalOriginal = parent()->rmGlobOriginalOriginal() * theRmGlobOriginalOriginal;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ SetOriginalEntryValues: Set orig coordinates and extra entry values for backup
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setOriginalEntryValues() 
{
  //---------- Set orig coordinates
  theCentreGlobOriginal = centreGlob();
  theRmGlobOriginal = rmGlob(); 

  theCentreGlobOriginalOriginal = centreGlob();
  theRmGlobOriginalOriginal = rmGlob(); 

  /*  if ( ALIUtils::debug >= 5 ) {
    ALIUtils::dump3v( centreGlob(), "OPTO: CENTRE GLOB ");
    ALIUtils::dumprm( rmGlob(), "OPTO: RM GLOB ");
    }*/
  
  //---------- Set extra entry values
  std::vector<ALIdouble>::const_iterator vdcite;
  for (vdcite = ExtraEntryValueList().begin(); 
       vdcite != ExtraEntryValueList().end(); vdcite++) {
      addExtraEntryValueOriginalToList( *vdcite );
      addExtraEntryValueOriginalOriginalToList( *vdcite );
  }
  //-  test();
  if( ALIUtils::debug >= 6 ) std::cout << " setOriginalEntryValues " << std::endl;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Propagate the light ray with the behaviour 'behav'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::participateInMeasurement(LightRay& lightray, Measurement& meas, const ALIstring& behav )
{
  //---------- see if light traverses or reflects
  setMeas( &meas );
  if ( behav == " " ) {
    defaultBehaviour( lightray, meas );
  } else if ( behav == "D" || behav == "DD" ) {
    detailedDeviatesLightRay( lightray );
  } else if ( behav == "T" || behav == "DT" ) {
    detailedTraversesLightRay( lightray );
  } else if ( behav == "FD" ) {
    fastDeviatesLightRay( lightray );
  } else if ( behav == "FT" ) {
    fastTraversesLightRay( lightray );
  } else if ( behav == "M" ) {
    makeMeasurement( lightray, meas );
  } else {
    userDefinedBehaviour( lightray, meas, behav);
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ default behaviour (depends of subclass type). A default behaviour can be makeMeasurement(), therefore you have to pass 'meas'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::defaultBehaviour( LightRay& lightray, Measurement& meas )
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement a default behaviour" << std::endl;
  std::cerr << " You have to specify some behaviour, like :D or :T or ..." << std::endl;
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of deviation of the light ray (reflection, shift, ...)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::fastDeviatesLightRay( LightRay& lightray ) 
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement deviation (:D)" << std::endl;
  std::cerr << " Please read documentation for this object type" << std::endl;
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of the light ray traversing
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::fastTraversesLightRay( LightRay& lightray )
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement the light traversing (:T)" << std::endl;
  std::cerr << " Please read documentation for this object type" << std::endl;
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of deviation of the light ray (reflection, shift, ...)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::detailedDeviatesLightRay( LightRay& lightray )
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement detailed deviation (:DD / :D)" << std::endl;
  std::cerr << " Please read documentation for this object type" << std::endl;
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of the light ray traversing
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::detailedTraversesLightRay( LightRay& lightray )
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement detailed traversing of light ray (:DT / :T)" << std::endl;
  std::cerr << " Please read documentation for this object type" << std::endl;
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Make the measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::makeMeasurement( LightRay& lightray, Measurement& meas )
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement making measurement (:M)" << std::endl;
  std::cerr << " Please read documentation for this object type" << std::endl;  
  exit(1);
}

void OpticalObject::userDefinedBehaviour( LightRay& lightray, Measurement& meas, const ALIstring& behav)
{
  std::cerr << "!!! Optical Object " << name() << " of type " << type() << " does not implement user defined behaviour = " << behav << std::endl;
  std::cerr << " Please read documentation for this object type" << std::endl;  
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get one of the plates of an OptO
//@@ 
//@@ The point is defined taking the centre of the splitter, 
//@@ and traslating it by +/-1/2 'width' in the direction of the splitter Z.
//@@ The normal of this plane is obtained as the splitter Z, 
//@@ and then it is rotated with the global rotation matrix. 
//@@ If applyWedge it is also rotated around the splitter X and Y axis by +/-1/2 of the wedge. 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIPlane OpticalObject::getPlate(const ALIbool forwardPlate, const ALIbool applyWedge) 
{
  if (ALIUtils::debug >= 4) std::cout << "% LR: GET PLATE " << name() << " forward= " << forwardPlate << std::endl;
  //---------- Get OptO variables
  const ALIdouble width = (findExtraEntryValue("width"));

  //---------- Get centre and normal of plate
  //----- Get plate normal before wedge (Z axis of OptO)
  CLHEP::Hep3Vector ZAxis(0.,0.,1.);
  CLHEP::HepRotation rmt = rmGlob();
  CLHEP::Hep3Vector plate_normal = rmt*ZAxis;

  //----- plate centre = OptO centre +/- 1/2 width before wedge
  CLHEP::Hep3Vector plate_point = centreGlob();
  //--- Add to it half of the width following the direction of the plate normal. -1/2 if it is forward plate, +1/2 if it is backward plate
  ALIdouble normal_sign = -forwardPlate*2 + 1;
  plate_point += normal_sign * width/2. * plate_normal;
  //-  if (ALIUtils::debug >= 4) std::cout << "width = " << width <<std::endl;
  if (ALIUtils::debug >= 3) {
    ALIUtils::dump3v( plate_point, "plate_point");
    ALIUtils::dump3v( plate_normal, "plate_normal before wedge");
    ALIUtils::dumprm( rmt, "rmt before wedge" );
  }

  if(applyWedge) {
    ALIdouble wedge;
    wedge = findExtraEntryValue("wedge");
    if( wedge != 0. ){
      //---------- Rotate plate normal by 1/2 wedge angles
      CLHEP::Hep3Vector XAxis(1.,0.,0.);
      XAxis = rmt*XAxis;
      plate_normal.rotate( normal_sign * wedge/2., XAxis );
      if (ALIUtils::debug >= 3) ALIUtils::dump3v( plate_normal, "plate_normal after wedgeX ");
      if (ALIUtils::debug >= 4) ALIUtils::dump3v( XAxis, "X Axis for applying wedge ");
      CLHEP::Hep3Vector YAxis(0.,1.,0.);
      YAxis = rmt*YAxis;
      plate_normal.rotate( normal_sign * wedge/2., YAxis );
      if (ALIUtils::debug >= 3) ALIUtils::dump3v( plate_normal, "plate_normal after wedgeY ");
      if (ALIUtils::debug >= 4) ALIUtils::dump3v( YAxis, "Y Axis for applying wedge ");
    }
  }

  //---------- Return plate plane
  return ALIPlane(plate_point, plate_normal);

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Displace the centre coordinate 'coor' to get the derivative 
//@@ of a measurement w.r.t this entry
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceCentreGlob( const XYZcoor coor, const ALIdouble disp) 
{
  if ( ALIUtils::debug >= 5 ) std::cout << name() << " displaceCentreGlob: coor " << coor << " disp = " << disp << std::endl;

  theCentreGlob = centreGlobOriginal(); 
  CLHEP::Hep3Vector dispVec = getDispVec( coor, disp ); 
  theCentreGlob += dispVec;

  //----------- Displace CentreGlob() of every component
  std::vector<OpticalObject*> vopto;
  ALIbool igetood = Model::getComponentOptOs(name(), vopto); 
  if( !igetood ) {
    //    std::cout << " NO MORE COMPONENTS IN THIS OptO" << name() << std::endl ; 
    return;
  }
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceCentreGlob( dispVec ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CLHEP::Hep3Vector OpticalObject::getDisplacementInLocalCoordinates( const XYZcoor coor, const ALIdouble disp )
{
  CLHEP::Hep3Vector dispVec;
  switch( coor ) {
  case 0:
    dispVec = CLHEP::Hep3Vector( disp, 0., 0. );
    break;
  case 1:
    dispVec = CLHEP::Hep3Vector( 0., disp, 0. );
    break;
  case 2:
    dispVec = CLHEP::Hep3Vector( 0., 0., disp );
    break;
  default:
    std::cerr << "!!! DISPLACECENTREGLOB coordinate should be 0-2, not " << coor << std::endl; 
    exit(2);
  }

  return dispVec;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Displace the centre coordinate 'coor' to get the derivative 
//@@ of a measurement w.r.t this entry
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceCentreGlob( const CLHEP::Hep3Vector& dispVec) 
{
  if ( ALIUtils::debug >= 5 ) std::cout << name() << " displaceCentreGlob: dispVec = " << dispVec << std::endl;

  theCentreGlob = centreGlobOriginal();
  theCentreGlob += dispVec; 

  //----------- Displace CentreGlob() of every component
  std::vector<OpticalObject*> vopto;
  ALIbool igetood = Model::getComponentOptOs(name(), vopto); 
  if( !igetood ) {
    return;
  }
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    (*vocite)->displaceCentreGlob( dispVec ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ displaceExtraEntry:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceExtraEntry(const ALIuint entryNo, const ALIdouble disp) 
{
  //  std::vector< ALIdouble >::iterator ite =  theExtraEntryValueVector.begin();
  ALIdouble Pentry_value = (*(theExtraEntryValueVector.begin() + entryNo));

  ALIdouble Pentry_orig_value = *(theExtraEntryValueOriginalVector.begin() + entryNo);
  Pentry_value = (Pentry_orig_value) + disp;
  //  std::cout << " displaceExtraEntry " << Pentry_value << " <> " << Pentry_orig_value << std::endl;
  theExtraEntryValueVector[entryNo] = Pentry_value;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setExtraEntryValue(const ALIuint entryNo, const ALIdouble val) 
{
  theExtraEntryValueVector[entryNo] = val;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceCentreGlobOriginal( const XYZcoor coor, const ALIdouble disp) 
{
  if ( ALIUtils::debug >= 4 ) std::cout << "@@ OpticalObject::displaceCentreGloboriginal " << name() << " " << coor << " " << disp << std::endl;
  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v(theCentreGlobOriginal, "the centre glob original 0");
  CLHEP::Hep3Vector dispVec = getDispVec( coor, disp ); 
  theCentreGlobOriginal += dispVec;

  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v(theCentreGlobOriginal, "the centre glob original displaced");
 
  //----------- Displace CentreGlob() of every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceCentreGlobOriginal( dispVec ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ displaceCentreGlobOriginal:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceCentreGlobOriginal( const CLHEP::Hep3Vector& dispVec) 
{
  if ( ALIUtils::debug >= 4 ) std::cout << " OpticalObject::displaceCentreGloboriginal " << name() << " dispVec " << dispVec << std::endl;

  theCentreGlobOriginal += dispVec;

  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v(theCentreGlobOriginal, "the centre glob original");
 
  //----------- Displace CentreGlob() of every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceCentreGlobOriginal( dispVec ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceCentreGlobOriginalOriginal( const XYZcoor coor, const ALIdouble disp) 
{
  if ( ALIUtils::debug >= 4 ) std::cout << "@@ OpticalObject::displaceCentreGloboriginal " << name() << " " << coor << " " << disp << std::endl;
  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v(theCentreGlobOriginalOriginal, "the centre glob originalOriginal 0");
  CLHEP::Hep3Vector dispVec = getDispVec( coor, disp ); 
  theCentreGlobOriginalOriginal += dispVec;

  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v(theCentreGlobOriginalOriginal, "the centre glob original displaced");
 
  //----------- Displace CentreGlob() of every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceCentreGlobOriginalOriginal( dispVec ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ displaceCentreGlobOriginal:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceCentreGlobOriginalOriginal( const CLHEP::Hep3Vector& dispVec) 
{
  if ( ALIUtils::debug >= 4 ) std::cout << " OpticalObject::displaceCentreGloboriginal " << name() << " dispVec " << dispVec << std::endl;

  theCentreGlobOriginalOriginal += dispVec;

  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v(theCentreGlobOriginalOriginal, "the centre glob original");
 
  //----------- Displace CentreGlob() of every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceCentreGlobOriginalOriginal( dispVec ); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Rotate around axis 'coor' to get the derivative of 
//@@ a measurement w.r.t this entry
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceRmGlobAroundGlobal( OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp) 
{
  if(ALIUtils::debug>=5) std::cout << name() << "DISPLACERMGLOBAROUNDGLOBAL" << coor << "disp" << disp << std::endl;
  //-------------------- Rotate rotation matrix
  theRmGlob = rmGlobOriginal();
  theCentreGlob = centreGlobOriginal();
  if(ALIUtils::debug >= 5 ) {
    std::cout << this->name() << std::endl;
    ALIUtils::dumprm( theRmGlob, "before disp rm " );
  }
  rotateItAroundGlobal( theRmGlob, coor, disp );
  if(ALIUtils::debug >= 5 ) {
    ALIUtils::dumprm( theRmGlob, "after disp rm " );
  }
  //-------------------- Rotation translate the centre of component OptO
  if(ALIUtils::debug >= 5) ALIUtils::dump3v( centreGlob(), " centre_glob before rotation" );
  if(ALIUtils::debug >= 5 ) ALIUtils::dump3v( centreGlobOriginal(), "         centreGlobOriginal before rotation" );
  if(opto1stRotated != this ) { //own _centre_glob is not displaced
    //---------- Distance to 1st rotated OptO
    CLHEP::Hep3Vector radiusOriginal =  centreGlobOriginal() - opto1stRotated->centreGlobOriginal();
    CLHEP::Hep3Vector radius_rotated = radiusOriginal;
    rotateItAroundGlobal( radius_rotated, coor, disp );
    theCentreGlob = centreGlobOriginal() + (radius_rotated - radiusOriginal);
    if(ALIUtils::debug >= 5) ALIUtils::dump3v( centreGlob(), " centre_glob after rotation" );
    if(ALIUtils::debug >= 5) ALIUtils::dump3v( centreGlobOriginal(), "         centre_globOriginal() after rotation" );
  }

  //----------- Displace every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    (*vocite)->displaceRmGlobAroundGlobal( opto1stRotated, coor, disp);
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Rotate around axis 'coor' to get the derivative of 
//@@ a measurement w.r.t this entry
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceRmGlobAroundLocal( OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp) 
{
  if( anglesIsGlobal ) {
    std::cerr << "!!!FATAL ERROR: angles in global coordinates not supported momentarily if 'rotateAroundGlobal' is set as a Global Option " << std::endl;
    abort();
  }

  if(ALIUtils::debug>=5) std::cout << name() << " DISPLACE_RMGLOB_AROUND_LOCAL " << coor << " disp " << disp << std::endl;
  //---------- Build the rmGlob and centreGlob again, with displacement values
  //----- Local rotation is build with entry values plus displacement
  theRmGlob = CLHEP::HepRotation();
  //---------- Set global rotation matrix
  //-------- Get rm from Entries
  if( coor == XCoor ) {
    theRmGlob.rotateX( getEntryRMangle(XCoor) + disp );
    if(ALIUtils::debug>=5) std::cout << " rmglob rotated around x " <<  getEntryRMangle(XCoor) + disp << std::endl;
  }else {
    theRmGlob.rotateX( getEntryRMangle(XCoor) );
  }
  if(ALIUtils::debug >= 4) {
    ALIUtils::dumprm( theRmGlob, "displaceRmGlobAroundLocal: rm local after X " );
  }

//-  std::cout << name() << " " << coor << " " << XCoor << " getEntryRMangle(coor) )" << getEntryRMangle(coor) << std::endl;
  if( coor == YCoor ) {
    theRmGlob.rotateY( getEntryRMangle(YCoor) + disp );
    if(ALIUtils::debug>=5) std::cout << " rmglob rotated around y " <<  getEntryRMangle(YCoor) + disp << std::endl;
  }else {
    theRmGlob.rotateY( getEntryRMangle(YCoor) );
  }
  if(ALIUtils::debug >= 4) {
    std::cout << "  getEntryRMangle(YCoor) " <<  getEntryRMangle(YCoor) << std::endl;
    ALIUtils::dumprm( theRmGlob, "displaceRmGlobAroundLocal: rm local after Y " );
  }

  if( coor == ZCoor ) {
    theRmGlob.rotateZ( getEntryRMangle(ZCoor) + disp );
    if(ALIUtils::debug>=5) std::cout << " rmglob rotated around z " <<  getEntryRMangle(ZCoor) + disp << std::endl;
  }else {
    theRmGlob.rotateZ( getEntryRMangle(ZCoor) );
  }
  if(ALIUtils::debug >= 4) {
    std::cout << "  getEntryRMangle(ZCoor) " << getEntryRMangle(ZCoor) << std::endl;
    ALIUtils::dumprm( theRmGlob, "SetRMLocalFromEntryValues: RM ");
  }

  //-  theCentreGlob = CLHEP::Hep3Vector(0.,0.,0.);
  if(ALIUtils::debug >= 5 && disp != 0) {
    std::cout << this->name() << std::endl;
    ALIUtils::dumprm( theRmGlob, "displaceRmGlobAroundLocal: rm local " );
  }


  if( !anglesIsGlobal ) {
    SetRMGlobFromRMLocal();
  }

  //----- calculate local rot axis with new rm glob
  calculateLocalRotationAxisInGlobal();

  //-  theCentreGlob = CLHEP::Hep3Vector(0.,0.,0.);
  if(ALIUtils::debug >= 5 && disp != 0) {
    std::cout << this->name() << std::endl;
    ALIUtils::dumprm( theRmGlob, "displaceRmGlobAroundLocal: rm global " );
  }

  if(opto1stRotated != this ) { //own _centre_glob doesn't rotate
    setGlobalCentre();
    if(ALIUtils::debug >= 5) {
      ALIUtils::dump3v( centreGlob(), " centre_glob after rotation" );
      ALIUtils::dump3v( centreGlobOriginal(), "         centre_globOriginal() after rotation" );
    }
  }

  //----------- Displace every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    (*vocite)->displaceRmGlobAroundLocal( opto1stRotated, coor, 0.);
    //for aroundglobal all components are explicitly rotated disp, for aroundLocal, they will be rotated automatically if the parent is rotated, as the rmGlob is built from scratch
  }
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::setGlobalCoordinatesOfComponents()
{

  // Calculate the displaced centreGlob and rmGlob of components
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    (*vocite)->setGlobalCoordinates();
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceRmGlobOriginal(const  OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp) 
{
  if(ALIUtils::debug>=9) std::cout << name() << " DISPLACEORIGRMGLOB " << coor << " disp " << disp << std::endl;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if(gomgr->GlobalOptions()["rotateAroundLocal"] == 0) {
    //-------------------- Rotate rotation matrix
    if( ALIUtils::debug >= 5 ) ALIUtils::dumprm(theRmGlobOriginal, (name() + ALIstring(" theRmGlobOriginal before displaced ")).c_str() );
    switch( coor ) {
    case 0:
      theRmGlobOriginal.rotateX( disp );
      break;
    case 1:
      theRmGlobOriginal.rotateY( disp );
      break;
    case 2:
      theRmGlobOriginal.rotateZ( disp );
      break;
    default:
      std::cerr << "!!! DISPLACERMGLOB coordinate should be 0-2, not " << coor << std::endl;
      exit(2);
    }


    //-------------------- Rotation translate the centre of component OptO
    if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlob(), "angles rotate centre_glob" );
    if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlobOriginal(), "         centreGlobOriginal" );
    if(opto1stRotated != this ) { //own _centre_glob doesn't rotate
      //---------- Distance to 1st rotated OptO
      CLHEP::Hep3Vector radiusOriginal =  centreGlobOriginal() - opto1stRotated->centreGlobOriginal();
      CLHEP::Hep3Vector radius_rotated = radiusOriginal;
      switch (coor) {
      case 0:
	radius_rotated.rotateX(disp);
	break;
      case 1:
	radius_rotated.rotateY(disp);
	break;
      case 2:
	radius_rotated.rotateZ(disp);
	break;
      default:
	break;  // already exited in previous switch
      }
      theCentreGlobOriginal = centreGlobOriginal() + (radius_rotated - radiusOriginal);
      if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlob(), "angle rotate centre_glob" );
      if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlobOriginal(), "         centre_globOriginal()" );
    }

    if( ALIUtils::debug >= 5 ) ALIUtils::dumprm(theRmGlobOriginal, (name() + ALIstring(" theRmGlobOriginal displaced ")).c_str() );
    
    //----------- Displace every OptO component
    std::vector<OpticalObject*> vopto;
    Model::getComponentOptOs(name(), vopto); 
    std::vector<OpticalObject*>::const_iterator vocite;
    for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceRmGlobOriginal( opto1stRotated, coor, disp);
    }

  } else {
    setGlobalCoordinates();
    theCentreGlobOriginal = theCentreGlob;
    theRmGlobOriginal = theRmGlob; //!!temporary, theRmGlobOriginal should disappear
    //----------- Displace every OptO component
    std::vector<OpticalObject*> vopto;
    Model::getComponentOptOs(name(), vopto); 
    std::vector<OpticalObject*>::const_iterator vocite;
    for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceRmGlobOriginal( opto1stRotated, coor, disp);
    }
    if( ALIUtils::debug >= 5 ) {
      ALIUtils::dump3v( theCentreGlob, " displaceRmGlobOriginal " );
      ALIUtils::dumprm( theRmGlob, " displaceRmGlobOriginal " );
    }
  }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceRmGlobOriginalOriginal(const  OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp) 
{
  if(ALIUtils::debug>=9) std::cout << name() << " DISPLACEORIGRMGLOB " << coor << " disp " << disp << std::endl;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if(gomgr->GlobalOptions()["rotateAroundLocal"] == 0) {
    //-------------------- Rotate rotation matrix
    if( ALIUtils::debug >= 5 ) ALIUtils::dumprm(theRmGlobOriginalOriginal, (name() + ALIstring(" theRmGlobOriginalOriginal before displaced ")).c_str() );
    switch( coor ) {
    case 0:
      theRmGlobOriginalOriginal.rotateX( disp );
      break;
    case 1:
      theRmGlobOriginalOriginal.rotateY( disp );
      break;
    case 2:
      theRmGlobOriginalOriginal.rotateZ( disp );
      break;
    default:
      std::cerr << "!!! DISPLACERMGLOB coordinate should be 0-2, not " << coor << std::endl;
      exit(2);
    }


    //-------------------- Rotation translate the centre of component OptO
    if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlob(), "angles rotate centre_glob" );
    if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlobOriginalOriginal(), "         centreGlobOriginalOriginal" );
    if(opto1stRotated != this ) { //own _centre_glob doesn't rotate
      //---------- Distance to 1st rotated OptO
      CLHEP::Hep3Vector radiusOriginalOriginal =  centreGlobOriginalOriginal() - opto1stRotated->centreGlobOriginalOriginal();
      CLHEP::Hep3Vector radius_rotated = radiusOriginalOriginal;
      switch (coor) {
      case 0:
	radius_rotated.rotateX(disp);
	break;
      case 1:
	radius_rotated.rotateY(disp);
	break;
      case 2:
	radius_rotated.rotateZ(disp);
	break;
      default:
	break;  // already exited in previous switch
      }
      theCentreGlobOriginalOriginal = centreGlobOriginalOriginal() + (radius_rotated - radiusOriginalOriginal);
      if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlob(), "angle rotate centre_glob" );
      if(ALIUtils::debug>=98)ALIUtils::dump3v( centreGlobOriginalOriginal(), "         centre_globOriginalOriginal()" );
    }

    if( ALIUtils::debug >= 5 ) ALIUtils::dumprm(theRmGlobOriginalOriginal, (name() + ALIstring(" theRmGlobOriginalOriginal displaced ")).c_str() );
    
    //----------- Displace every OptO component
    std::vector<OpticalObject*> vopto;
    Model::getComponentOptOs(name(), vopto); 
    std::vector<OpticalObject*>::const_iterator vocite;
    for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceRmGlobOriginalOriginal( opto1stRotated, coor, disp);
    }

  } else {
    setGlobalCoordinates();
    theCentreGlobOriginalOriginal = theCentreGlob;
    theRmGlobOriginalOriginal = theRmGlob; //!!temporary, theRmGlobOriginalOriginal should disappear
    //----------- Displace every OptO component
    std::vector<OpticalObject*> vopto;
    Model::getComponentOptOs(name(), vopto); 
    std::vector<OpticalObject*>::const_iterator vocite;
    for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
      (*vocite)->displaceRmGlobOriginalOriginal( opto1stRotated, coor, disp);
    }
    if( ALIUtils::debug >= 5 ) {
      ALIUtils::dump3v( theCentreGlob, " displaceRmGlobOriginalOriginal " );
      ALIUtils::dumprm( theRmGlob, " displaceRmGlobOriginalOriginal " );
    }
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceExtraEntryOriginal( const ALIuint entryNo, const ALIdouble disp) 
{
  ALIdouble Pentry_orig_value = *(theExtraEntryValueOriginalVector.begin() + entryNo);
  Pentry_orig_value += disp;
  //  std::cout << " displaceExtraEntryOriginal " << *(theExtraEntryValueOriginalVector.begin() + entryNo) << " + " << disp << std::endl;
  theExtraEntryValueOriginalVector[entryNo] = Pentry_orig_value;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::displaceExtraEntryOriginalOriginal( const ALIuint entryNo, const ALIdouble disp) 
{
  ALIdouble Pentry_orig_value = *(theExtraEntryValueOriginalOriginalVector.begin() + entryNo);
  Pentry_orig_value += disp;
  //  std::cout << " displaceExtraEntryOriginalOriginal " << *(theExtraEntryValueOriginalOriginalVector.begin() + entryNo) << " + " << disp << std::endl;
  theExtraEntryValueOriginalOriginalVector[entryNo] = Pentry_orig_value;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const ALIint OpticalObject::extraEntryNo( const ALIstring& entry_name ) const
{
  //-  std::cout << ExtraEntryList().size() << "entry name " << entry_name << std::endl;

  std::vector<Entry*>::const_iterator vecite;
  for (vecite = ExtraEntryList().begin(); vecite != ExtraEntryList().end(); vecite++) {
    //-    std::cout <<"in entryno" << (*vecite)->name() << entry_name << std::endl; 
    if ((*vecite)->name() == entry_name ) {
      return (vecite - ExtraEntryList().begin());
    } 
    //-    std::cout <<"DD in entryno" << (*vecite)->name() << entry_name << std::endl; 
  }
  //-  std::cout << "!!: extra entry name not found: " << entry_name << " in OptO " << name() << std::endl;
  //  exit(2);
  return ALIint(-1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Find an extra Entry by name and return its value. If entry not found, return 0.
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const ALIdouble OpticalObject::findExtraEntryValue( const ALIstring& eename ) const {
  ALIdouble retval; 
  const ALIint entryNo = extraEntryNo( eename );
  if( entryNo >= 0 ) {
    const ALIdouble Pentry_value = *(theExtraEntryValueVector.begin() + entryNo);
    retval = (Pentry_value);
  } else {
    //    if(ALIUtils::debug >= 0) std::cerr << "!!Warning: entry not found; " << eename << ", in object " << name() << " returns 0. " << std::endl;
    ALIdouble check;
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    gomgr->getGlobalOptionValue("check_extra_entries", check );
    if( check == 1) {
    //    if( check <= 1) {//exit temporarily
      std::cerr << "!!OpticalObject:ERROR: entry not found; " << eename << ", in object " << name() << std::endl;
      exit(1); 
    } else {
      //-      std::cerr << "!!temporal WARNING in OpticalObject::findExtraEntryValue: entry not found; " << eename << ", in object " << name() << std::endl;
      retval = 0.;
    }
  }

  if(ALIUtils::debug >= 5)  std::cout << " OpticalObject::findExtraEntryValue: " << eename << " = " << retval << std::endl;
  return retval;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Find an extra Entry by name and return its value. If entry not found, stop.
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const ALIdouble OpticalObject::findExtraEntryValueMustExist( const ALIstring& eename ) const 
{
  ALIdouble entry = findExtraEntryValue( eename );
  const ALIint entryNo = extraEntryNo( eename );
  if( entryNo < 0) {
    std::cerr << "!!OpticalObject::findExtraEntryValueMustExist: ERROR: entry not found; " << eename << ", in object " << name() << std::endl;
    exit(1); 
  }
  //  if(ALIUtils::debug >= 5)  std::cout << " OpticalObject::findExtraEntryValueMustExist: " << eename << " = " << entry << std::endl;
  return entry;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Find an extra Entry by name and pass its value. Return if entry is found or not
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const ALIbool OpticalObject::findExtraEntryValueIfExists( const ALIstring& eename, ALIdouble& value ) const 
{
  value = findExtraEntryValue( eename );
  const ALIint entryNo = extraEntryNo( eename );
  //-    std::cout << eename << " entryNo " << entryNo << " value " << value << std::endl;
  return( entryNo >= 0 );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ resetGlobalCoordinates: Reset Global Coordinates (after derivative is finished)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::resetGlobalCoordinates()
{

  //---------- Reset centre and rm
  theRmGlob = rmGlobOriginal();
  theCentreGlob = centreGlobOriginal();

  //---------- Reset extra entries 
  //---------- Set extra entry values list
  std::vector<ALIdouble>::iterator vdite;
  std::vector<ALIdouble>::const_iterator vdcite_o = ExtraEntryValueOriginalList().begin() ;
  for (vdite = ExtraEntryValueList().begin(); 
       vdite != ExtraEntryValueList().end(); vdite++,vdcite_o++) {
    (*vdite) = (*vdcite_o);
  }

  //----------- Reset entries of every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    (*vocite)->resetGlobalCoordinates();
  }

  calculateLocalRotationAxisInGlobal();
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ resetGlobalCoordinates: Reset Global Coordinates (after derivative is finished)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::resetOriginalOriginalCoordinates()
{
  //  std::cout << " !!! CALLING resetOriginalOriginalCoordinates(). STOP " << std::endl;

  //---------- Reset centre and rm
  theRmGlob = theRmGlobOriginalOriginal;
  theCentreGlob = theCentreGlobOriginalOriginal;
  theRmGlobOriginal = theRmGlobOriginalOriginal;
  theCentreGlobOriginal = theCentreGlobOriginalOriginal;


  //---------- Reset extra entry values list
  std::vector<ALIdouble>::iterator vdite;
  std::vector<ALIdouble>::iterator vdite_o = theExtraEntryValueOriginalVector.begin() ;
  std::vector<ALIdouble>::const_iterator vdcite_oo = theExtraEntryValueOriginalOriginalVector.begin() ;
  std::vector<Entry*>::const_iterator vdciteE = ExtraEntryList().begin() ;
  for (vdite = ExtraEntryValueList().begin(); 
       vdite != ExtraEntryValueList().end(); vdite++,vdite_o++,vdcite_oo++,vdciteE++) {
    (*vdite) = (*vdcite_oo);
    (*vdite_o) = (*vdcite_oo); 
    (*vdciteE)->addFittedDisplacementToValue( - (*vdciteE)->valueDisplacementByFitting() );
    //      std::cout << " resetting extra entry origorig " << (*vdciteE)->name() << " = " << (*vdite) << " ? " << (*vdcite_oo)  << std::endl;
      //      std::cout << " resetting extra entry origorig " << (*vdciteE)->name() << " = " << (*vdite) << " ? " << (*vdciteE)->value()  << std::endl;
    //  std::cout << " check extra entry " << (*vdciteE)->value() << " =? " << (*vdite) << std::endl;
  }

  /*  std::vector< Entry* >::iterator eite;
  for( eite = theCoordinateEntryVector.begin(); eite != theCoordinateEntryVector.end(); eite++ ){
    (*eite)->addFittedDisplacementToValue( - (*eite)->valueDisplacementByFitting() );
  }
  */


  calculateLocalRotationAxisInGlobal();

  //----------- Reset entries of every component
  std::vector<OpticalObject*> vopto;
  Model::getComponentOptOs(name(), vopto); 
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); vocite++) {
    (*vocite)->resetOriginalOriginalCoordinates();
  }


}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Destructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalObject::~OpticalObject()
{
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Return the name of the OptO without its path
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const ALIstring OpticalObject::shortName() const
{
  ALIint last_slash = name().rfind('/');
  ALIstring sname = name().substr(last_slash+1, name().size()-1);
  if( last_slash == -1 ) { //object of type "system"
    sname = name();
  } else {
    sname = name().substr(last_slash+1, name().size()-1);
  }
  return sname; 
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::ostream& operator << (std::ostream& os, const OpticalObject& c) {
  os << "OPTICALOBJECT: " << c.theName << " of type: " << c.theType 
     << "  " << c.theCentreGlob 
     << c.theRmGlob << std::endl;

  return os;

}
  // Lists of entries
//  std::vector< Entry* > theCoordinateEntryVector;
//  std::vector< Entry* > theExtraEntryVector;


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::vector<double> OpticalObject::GetLocalRotationAngles( std::vector< Entry* > entries )
{
  CLHEP::HepRotation rmParent = theParent->rmGlob();  //ORIGINAL ?????????????????
  CLHEP::HepRotation rmLocal = rmParent.inverse() * theRmGlob;
     
  //I was using theRmGlobOriginal, assuming it has been set to theRmGlob already, check it, in case it may have other consequences
  if( theRmGlobOriginal != theRmGlob ){
    std::cerr << " !!!FATAL ERROR: OpticalObject::GetLocalRotationAngles   theRmGlobOriginal != theRmGlob " << std::endl;
    ALIUtils::dumprm( theRmGlobOriginal, " theRmGlobOriginal ");
    ALIUtils::dumprm( theRmGlob, " theRmGlob ");
    exit(1);
  }

  if( ALIUtils::debug >= 5 ) {
    std::cout << " OpticalObject::GetLocalRotationAngles " << name() << " theParent " << theParent->name() << std::endl;
    ALIUtils::dumprm( rmParent, " rm parent ");
    ALIUtils::dumprm( rmLocal, " rm local ");
    ALIUtils::dumprm( theRmGlobOriginal, " theRmGlobOriginal ");
    ALIUtils::dumprm( theRmGlob, " theRmGlob ");
  }
  return GetRotationAnglesFromMatrix( rmLocal, entries );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::vector<double> OpticalObject::GetRotationAnglesFromMatrix( CLHEP::HepRotation& rmLocal, std::vector< Entry* > entries )
{
  double pii = acos(0.)*2;
  std::vector<double> newang(3);
  double angleX = entries[3]->value()+entries[3]->valueDisplacementByFitting();
  double angleY = entries[4]->value()+entries[4]->valueDisplacementByFitting();
  double angleZ = entries[5]->value()+entries[5]->valueDisplacementByFitting();

  if( ALIUtils::debug >= 4 ) {
    std::cout << " angles as value entries: X= " << angleX << " Y= " << angleY << " Z " << angleZ << std::endl;
  }

  //-  std::cout << name () << " vdbf " << angleX << " " << angleY << " " << angleZ << std::endl;
  double rotzx = approxTo0( rmLocal.zx() );
  double rotzy = approxTo0( rmLocal.zy() );
  double rotzz = approxTo0( rmLocal.zz() );
  double rotyx = approxTo0( rmLocal.yx() );
  double rotxx = approxTo0( rmLocal.xx() );
  if( rotzy == 0. && rotzz == 0. ) {
    //check that entry is z angle
    newang[0] = angleX;
    //beware of aa <==> pii - aa
    if( eq2ang( rmLocal.zx(), -1. ) ) {
      double aa = asin( rmLocal.xy() );
      if( diff2pi( angleZ, - aa + newang[0] ) < diff2pi( angleZ, - pii + aa + newang[0] )  ) {
	newang[2] = -aa + newang[0];
	if( ALIUtils::debug >= 5 ) std::cout << " newang[0] = -aa + newang[0] " << std::endl;
      } else {
	newang[2] = -pii + aa + newang[0];
	if( ALIUtils::debug >= 5 ) std::cout << " newang[0] = -pii + aa + newang[0] " << newang[0] << " " << aa << " " << newang[2] << std::endl;
      }
    } else {
      double aa = asin( -rmLocal.xy() );
      if( diff2pi( angleZ, aa - newang[0] ) < diff2pi( angleZ, pii - aa - newang[0] )  ) {
	newang[2] = aa - newang[0];
	if( ALIUtils::debug >= 5 ) std::cout << " newang[0] = aa - newang[2] " << std::endl;
      } else {
	newang[2] = pii - aa - newang[0];
	if( ALIUtils::debug >= 5 ) std::cout << " newang[0] = pii - aa - newang[2] " << newang[0] << " " << aa << " " << newang[2] << std::endl;
      }
    } 
  } else {
    newang[0] = atan( rotzy / rotzz );
    newang[2] = atan( rotyx / rotxx );
  }
  if( rotzx < -1. ) {
    //-    std::cerr << " rotzx too small " << rotzx << " = " << rmLocal.zx() << " " << rotzx-rmLocal.zx() << std::endl;
    rotzx = -1.;
  } else if( rotzx > 1. ) {
    //-    std::cerr << " rotzx too big " << rotzx << " = " << rmLocal.zx() << " " << rotzx-rmLocal.zx() << std::endl;
    rotzx = 1.;
  }
  newang[1] = -asin( rotzx );
  if( ALIUtils::debug >= 5 ) std::cout << "First calculation of angles: " << std::endl 
			       << " newang[0] " << newang[0] << " " << rotzy << " " << rotzz << std::endl
			       << " newang[1] " << newang[1] << " " << rotzx << std::endl
			       << " newang[2] " << newang[2] << " " << rotyx << " " << rotxx << std::endl;
  
  //    newang[2] = acos( rmLocal.xx() / cos( newang[1] ) );
  //----- CHECK if the angles are OK (there are several symmetries)
  //--- Check if the predictions with the angles obtained match the values of the rotation matrix (they may differ for exampole by a sign or more in complicated formulas)
  double rotnewxx = cos( newang[1] ) * cos( newang[2] );
  double rotnewzz = cos( newang[0] ) * cos( newang[1] );
  double rotnewxy = sin( newang[0] ) * sin( newang[1] ) * cos( newang[2] ) - cos( newang[0] )* sin( newang[2] );
  double rotnewxz = cos( newang[0] ) * sin( newang[1] ) * cos( newang[2] ) + sin( newang[0] )* sin( newang[2] );
  double rotnewyy = sin( newang[0] ) * sin( newang[1] ) * sin( newang[2] ) + cos( newang[0] )* cos( newang[2] );
  double rotnewyz = cos( newang[0] ) * sin( newang[1] ) * sin( newang[2] ) - sin( newang[0] )* cos( newang[2] );

  bool eqxx = eq2ang( rotnewxx, rmLocal.xx() );
  bool eqzz = eq2ang( rotnewzz, rmLocal.zz() );
  bool eqxy = eq2ang( rotnewxy, rmLocal.xy() );
  bool eqxz = eq2ang( rotnewxz, rmLocal.xz() );
  bool eqyy = eq2ang( rotnewyy, rmLocal.yy() );
  bool eqyz = eq2ang( rotnewyz, rmLocal.yz() );

  //--- Check if one of the tree angles should be changed
  if( ALIUtils::debug >= 5 ) {
    std::cout << " pred rm.xx " << rotnewxx << " =? " << rmLocal.xx() 
	 << " pred rm.zz " << rotnewzz << " =? " << rmLocal.zz() 
	 << std::endl;
    std::cout << " eqxx " << eqxx << " eqzz " << eqzz << std::endl;
    //-    std::cout << " rotnewxx " << rotnewxx << " = " << rmLocal.xx() << " " << fabs( rotnewxx - rmLocal.xx() ) << " " <<(fabs( rotnewxx - rmLocal.xx() ) < 0.0001) << std::endl;
  }

  if( eqxx & !eqzz ) {
    newang[0] = pii + newang[0];
    if( ALIUtils::debug >= 5 ) std::cout << " change newang[0] " << newang[0] << std::endl;
  } else  if( !eqxx & !eqzz ) {
    newang[1] = pii - newang[1];
    if( ALIUtils::debug >= 5 ) std::cout << " change newang[1] " << newang[1] << std::endl;
  } else  if( !eqxx & eqzz ) {
    newang[2] = pii + newang[2];
    if( ALIUtils::debug >= 5 ) std::cout << " change newang[2] " << newang[2] << std::endl;
  }

  //--- Check if the 3 angles should be changed (previous check is invariant to the 3 changing)
  if( ALIUtils::debug >= 5 ) {
    std::cout << " pred rm.xy " << rotnewxy << " =? " << rmLocal.xy() 
	 << " pred rm.xz " << rotnewxz << " =? " << rmLocal.xz() 
	 << " pred rm.yy " << rotnewyy << " =? " << rmLocal.yy()
	 << " pred rm.yz " << rotnewyz << " =? " << rmLocal.yz()
	 << std::endl;
    std::cout << " eqxy " << eqxy << " eqxz " << eqxz << " eqyy " << eqyy << " eqyz " << eqyz << std::endl;
  }

  if( !eqxy || !eqxz || !eqyy || !eqyz ) {
    // check also cases where one of the above 'eq' is OK because it is = 0
    if( ALIUtils::debug >= 5 ) std::cout << " change the 3 newang " << std::endl;
    newang[0] = addPii( newang[0] );
    newang[1] = pii - newang[1];
    newang[2] = addPii( newang[2] );
    double rotnewxy = -sin( newang[0] ) * sin( newang[1] ) * cos( newang[2] ) - cos( newang[0] )* sin( newang[2] );
    double rotnewxz = -cos( newang[0] ) * sin( newang[1] ) * cos( newang[2] ) - sin( newang[0] )* sin( newang[2] );
    if( ALIUtils::debug >= 5 ) std::cout << " rotnewxy " << rotnewxy << " = " << rmLocal.xy()
	 << " rotnewxz " << rotnewxz << " = " << rmLocal.xz() << std::endl;
  }
  if( diff2pi(angleX, newang[0] ) + diff2pi(angleY, newang[1] ) +diff2pi(angleZ, newang[2] )
	   > diff2pi(angleX, pii+newang[0] ) + diff2pi(angleY, pii-newang[1] ) + diff2pi(angleZ, pii+newang[2] ) ){
    // check also cases where one of the above 'eq' is OK because it is = 0
    if( ALIUtils::debug >= 5 ) std::cout << " change the 3 newang " << std::endl;
    newang[0] = addPii( newang[0] );
    newang[1] = pii - newang[1];
    newang[2] = addPii( newang[2] );
    double rotnewxy = -sin( newang[0] ) * sin( newang[1] ) * cos( newang[2] ) - cos( newang[0] )* sin( newang[2] );
    double rotnewxz = -cos( newang[0] ) * sin( newang[1] ) * cos( newang[2] ) - sin( newang[0] )* sin( newang[2] );
    if( ALIUtils::debug >= 5 ) std::cout << " rotnewxy " << rotnewxy << " = " << rmLocal.xy()
	 << " rotnewxz " << rotnewxz << " = " << rmLocal.xz() << std::endl;
  }
  
  for (int ii=0; ii<3; ii++) {  
    newang[ii] = approxTo0( newang[ii] );
  }
  //  double rotnewyx = cos( newang[1] ) * sin( newang[2] );

  if(  checkMatrixEquations( newang[0], newang[1], newang[2], &rmLocal ) != 0 ){
    std::cerr << " wrong rotation matrix " <<  newang[0] << " " << newang[1] << " " << newang[2] << std::endl;
    ALIUtils::dumprm( rmLocal, " matrix is " );
  }
  if( ALIUtils::debug >= 5 ) {
    std::cout << "Final angles:  newang[0] " << newang[0] << " newang[1] " << newang[1] << " newang[2] " << newang[2] << std::endl;
    CLHEP::HepRotation rot;
    rot.rotateX( newang[0] );
    ALIUtils::dumprm( rot, " new rot after X ");
    rot.rotateY( newang[1] );
    ALIUtils::dumprm( rot, " new rot after Y ");
    rot.rotateZ( newang[2] );
    ALIUtils::dumprm( rot, " new rot ");
    ALIUtils::dumprm( rmLocal, " rmLocal " );
    //-    ALIUtils::dumprm( theRmGlobOriginal, " theRmGlobOriginal " );
  }

  //-  std::cout << " before return newang[0] " << newang[0] << " newang[1] " << newang[1] << " newang[2] " << newang[2] << std::endl;
  return newang;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::calculateLocalRotationAxisInGlobal()
{
  axisXLocalInGlobal = CLHEP::Hep3Vector(1.,0.,0.);
  axisXLocalInGlobal *= theRmGlob;
  axisYLocalInGlobal = CLHEP::Hep3Vector(0.,1.,0.);
  axisYLocalInGlobal *= theRmGlob;
  axisZLocalInGlobal = CLHEP::Hep3Vector(0.,0.,1.);
  axisZLocalInGlobal *= theRmGlob;
  if( ALIUtils::debug >= 4 ){
    std::cout << name() << " axis X local in global " << axisXLocalInGlobal << std::endl;
    std::cout << name() << " axis Y local in global " << axisYLocalInGlobal << std::endl;
    std::cout << name() << " axis Z local in global " << axisZLocalInGlobal << std::endl;
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
template<class T>
void OpticalObject::rotateItAroundGlobal( T& object, const XYZcoor coor, const double disp )
{
  switch (coor) {
    case 0:
      object.rotateX(disp);
      break;
    case 1:
      object.rotateY(disp);
      break;
    case 2:
      object.rotateZ(disp);
      break;
  }
      //  CLHEP::Hep3Vector axisToRotate = GetAxisForDisplacement( coor );
      //  object.rotate(disp, axisToRotate);
  if( ALIUtils::debug >= 5 ) std::cout << " rotateItAroundGlobal coor " << coor << " disp " << disp << std::endl;
}


/*
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CLHEP::Hep3Vector OpticalObject::GetAxisForDisplacement( const XYZcoor coor )
{
  CLHEP::Hep3Vector axis;
  if(Model::GlobalOptions()["rotateAroundLocal"] == 0) {
    switch (coor) {
    case 0:
      axis = CLHEP::Hep3Vector( 1., 0., 0. );
      break;
    case 1:
      axis = CLHEP::Hep3Vector( 0., 1., 0. );
      break;
    case 2:
      axis = CLHEP::Hep3Vector( 0., 0., 1. );
      break;
    default:
      break;  // already exited in previous switch
    }
  } else {
    switch (coor) {
    case 0:
      if( ALIUtils::debug >= 5 ) std::cout << coor << "rotate local " << axisXLocalInGlobal << std::endl;
      axis = axisXLocalInGlobal;
      break;
    case 1:
      if( ALIUtils::debug >= 5 ) std::cout << coor << "rotate local " << axisLocalInGlobal << std::endl;
      axis = axisYLocalInGlobal;
      break;
    case 2:
       if( ALIUtils::debug >= 5 ) std::cout << coor << "rotate local " << axisZLocalInGlobal << std::endl;
      axis = axisZLocalInGlobal;
      break;
    default:
      break;  // already exited in previous switch
    }
  }

   return axis;
}
*/

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
double OpticalObject::diff2pi( double ang1, double ang2 ) 
{
  double pii = acos(0.)*2;
  double diff = fabs( ang1 - ang2 );
  diff = diff - int(diff/2./pii) * 2 *pii;
  return diff;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
bool OpticalObject::eq2ang( double ang1, double ang2 ) 
{
  bool beq;

  double pii = acos(0.)*2;
  double diff = diff2pi( ang1, ang2 );
  if( diff > 0.00001 ) {
    if( fabs( diff - 2*pii ) > 0.00001 ) {
      //-      std::cout << " diff " << diff << " " << ang1 << " " << ang2 << std::endl;
      beq = false;
    }
  } else {
    beq = true;
  }

  return beq;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
double OpticalObject::approxTo0( double val )
{
  double precision = 1.e-9;
  if( fabs(val) < precision ) val = 0;
  return val;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
double OpticalObject::addPii( double val )
{
  if( val < M_PI ) {
    val += M_PI;
  } else {
    val -= M_PI;
  }

  return val;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
int OpticalObject::checkMatrixEquations( double angleX, double angleY, double angleZ, CLHEP::HepRotation* rot)
{
  //-  std::cout << " cme " << angleX << " " << angleY << " " << angleZ << std::endl;
  if( rot == 0 ) {
    rot = new CLHEP::HepRotation();
    rot->rotateX( angleX );
    rot->rotateY( angleY );
    rot->rotateZ( angleZ );
  }
  double sx = sin(angleX);
  double cx = cos(angleX);
  double sy = sin(angleY);
  double cy = cos(angleY);
  double sz = sin(angleZ);
  double cz = cos(angleZ);

  double rotxx = cy*cz;
  double rotxy = sx*sy*cz-cx*sz;
  double rotxz = cx*sy*cz+sx*sz;
  double rotyx = cy*sz;
  double rotyy = sx*sy*sz+cx*cz;
  double rotyz = cx*sy*sz-sx*cz;
  double rotzx = -sy;
  double rotzy = sx*cy;
  double rotzz = cx*cy;

  int matrixElemBad = 0; 
  if( !eq2ang( rot->xx(), rotxx ) ) {
    std::cerr << " EQUATION for xx() IS BAD " << rot->xx() << " <> " << rotxx << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->xy(), rotxy ) ) {
    std::cerr << " EQUATION for xy() IS BAD " << rot->xy() << " <> " << rotxy << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->xz(), rotxz ) ) {
    std::cerr << " EQUATION for xz() IS BAD " << rot->xz() << " <> " << rotxz << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->yx(), rotyx ) ) {
    std::cerr << " EQUATION for yx() IS BAD " << rot->yx() << " <> " << rotyx << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->yy(), rotyy ) ) {
    std::cerr << " EQUATION for yy() IS BAD " << rot->yy() << " <> " << rotyy << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->yz(), rotyz ) ) {
    std::cerr << " EQUATION for yz() IS BAD " << rot->yz() << " <> " << rotyz << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->zx(), rotzx ) ) {
    std::cerr << " EQUATION for zx() IS BAD " << rot->zx() << " <> " << rotzx << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->zy(), rotzy ) ) {
    std::cerr << " EQUATION for zy() IS BAD " << rot->zy() << " <> " << rotzy << std::endl;
    matrixElemBad++;
  }
  if( !eq2ang( rot->zz(), rotzz ) ) {
    std::cerr << " EQUATION for zz() IS BAD " << rot->zz() << " <> " << rotzz << std::endl;
    matrixElemBad++;
  }

  //-  std::cout << " cme: matrixElemBad " << matrixElemBad << std::endl;
  return matrixElemBad;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CLHEP::Hep3Vector OpticalObject::getDispVec( const XYZcoor coor, const ALIdouble disp)
{
  CLHEP::Hep3Vector dispVec;
  switch (coor) {
  case 0:
    dispVec = CLHEP::Hep3Vector( disp, 0., 0. );
    break;
  case 1:
    dispVec = CLHEP::Hep3Vector( 0., disp, 0. );
    break;
  case 2:
    dispVec = CLHEP::Hep3Vector( 0., 0., disp );
    break;
  default:
    break;  // already exited in previous switch
  }
  //-  CLHEP::Hep3Vector dispVec = getDisplacementInLocalCoordinates( coor, disp); 
  if ( ALIUtils::debug >= 5 ) {
    ALIUtils::dump3v( dispVec, " dispVec in local " );
    CLHEP::HepRotation rmt = parent()->rmGlob();
    ALIUtils::dumprm( rmt, "parent rmGlob ");
  }
  dispVec = parent()->rmGlob() * dispVec;
  if ( ALIUtils::debug >= 5 ) ALIUtils::dump3v( dispVec, " dispVec in global " );

  return dispVec;
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const CLHEP::Hep3Vector OpticalObject::centreLocal() const 
{

  CLHEP::Hep3Vector cLocal = theCentreGlob -  parent()->centreGlob();
  CLHEP::HepRotation rmParentInv = inverseOf( parent()->rmGlob() );
  cLocal = rmParentInv * cLocal;

  return cLocal;
  /*-

  if( theCoordinateEntryVector.size() >= 3 ) {
    return CLHEP::Hep3Vector( theCoordinateEntryVector[0]->value(), theCoordinateEntryVector[1]->value(), theCoordinateEntryVector[2]->value() );
  } else {
    return CLHEP::Hep3Vector(0.,0.,0.);
  }
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const double OpticalObject::getEntryCentre( const XYZcoor coor ) const
{
  Entry* ce = theCoordinateEntryVector[coor];
  //  std::cout << coor << " getEntryCentre " << ce->value() << " + " << ce->valueDisplacementByFitting() << std::endl; 
  return ce->value() + ce->valueDisplacementByFitting();
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const double OpticalObject::getEntryCentre( const ALIstring& coorstr ) const
{
  XYZcoor coor;
  if( coorstr == "X" ) {
    coor = XCoor;
  }else if( coorstr == "Y" ) {
    coor = YCoor;
  }else  if( coorstr == "Z" ) {
    coor = ZCoor;
  }
  Entry* ce = theCoordinateEntryVector[coor];
  //  std::cout << coor << " getEntryCentre " << ce->value() << " + " << ce->valueDisplacementByFitting() << std::endl; 
  return ce->value() + ce->valueDisplacementByFitting();
} 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const double OpticalObject::getEntryRMangle( const XYZcoor coor ) const{
  Entry* ce = theCoordinateEntryVector[coor+3];
  //  std::cout << coor << " getEntryRMangle " << ce->value() << " + " << ce->valueDisplacementByFitting() << " size = " << theCoordinateEntryVector.size() << " ce = " << ce << " entry name " << ce->name() << " opto name " << name() << std::endl; 
  
  return ce->value() + ce->valueDisplacementByFitting();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const double OpticalObject::getEntryRMangle(  const ALIstring& coorstr ) const
{
  XYZcoor coor;
  if( coorstr == "X" ) {
    coor = XCoor;
  }else if( coorstr == "Y" ) {
    coor = YCoor;
  }else  if( coorstr == "Z" ) {
    coor = ZCoor;
  }
  Entry* ce = theCoordinateEntryVector[coor+3];
  //  std::cout << coor << " getEntryRMangle " << ce->value() << " + " << ce->valueDisplacementByFitting() << " size = " << theCoordinateEntryVector.size() << " ce = " << ce << " entry name " << ce->name() << " opto name " << name() << std::endl; 
  
  return ce->value() + ce->valueDisplacementByFitting();
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::constructMaterial()
{

  theMaterial =  new CocoaMaterialElementary( "Hydrogen", 1.*g/cm3, "H", 1 , 1 );

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OpticalObject::constructSolidShape()
{
  theSolidShape = new CocoaSolidShapeBox( "Box", 10.*cm, 10.*cm, 10.*cm );
}
