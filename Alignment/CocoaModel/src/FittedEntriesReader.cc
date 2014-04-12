#include "Alignment/CocoaModel/interface/FittedEntriesReader.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/ALIRmDataFromFile.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
  

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
FittedEntriesReader::FittedEntriesReader( const ALIstring& filename ) 
{
  theFileName = filename;
  theFile = ALIFileIn::getInstance( filename );
  std::vector<ALIstring> wl;
  theFile.getWordsInLine( wl );
  if (wl[0] == ALIstring( "DIMENSIONS:" ) ){
    theLengthDim = ALIUtils::CalculateLengthDimensionFactorFromString( wl[3] );
    theLengthErrorDim = ALIUtils::CalculateLengthDimensionFactorFromString( wl[5] );
    theAngleDim = ALIUtils::CalculateAngleDimensionFactorFromString( wl[8] );
    theAngleErrorDim = ALIUtils::CalculateAngleDimensionFactorFromString( wl[10] );
  } else {
    ALIUtils::dumpVS( wl, "!!! FATAL ERROR FittedEntriesReader: first line is not dimensions " );
    std::exception();
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool FittedEntriesReader::readFittedEntriesFromFile()
{
  if( ALIUtils::debug >= 5) std::cout << " readFittedEntriesFromFile " << theFileName << std::endl;
  std::map<OpticalObject*,ALIRmDataFromFile> affAngles;

  std::vector<ALIstring> wl;
  theFile.getWordsInLine( wl );
  unsigned int siz = wl.size();
  for( size_t ii = 1; ii < siz; ii+=3 ){
    ALIstring optOentryName = substitutePointBySlash( wl[ii] );
    Entry* entry = Model::getEntryByName( optOentryName );
    if( ALIUtils::debug >= 5) std::cout << entry->name() << " readFittedEntriesFromFile " << entry->value() << " " << ALIUtils::getFloat( wl[ii+1] ) << std::endl;
    if( entry->name().substr(0,6) != "angles" ) {
      entry->displaceOriginalOriginal(entry->value() - ALIUtils::getFloat( wl[ii+1] )*theLengthDim);
    } else {
      OpticalObject* opto = entry->OptOCurrent();
      if( affAngles.find(opto) == affAngles.end() ){
       	affAngles[opto] = ALIRmDataFromFile();
      }
      std::map<OpticalObject*,ALIRmDataFromFile>::iterator ite = affAngles.find(opto);
      (*ite).second.setAngle( optOentryName.substr(optOentryName.size()-1,1), ALIUtils::getFloat( wl[ii+1] )*theAngleDim );
      if( ALIUtils::debug >= 5) std::cout << " setting angle from file " << ALIUtils::getFloat( wl[ii+1] )*theAngleDim << " " << wl[ii+1] << " " << theAngleDim << std::endl;
    }
    entry->setSigma( ALIUtils::getFloat( wl[ii+2] )*theAngleDim );
    //  ar.lass6.laser.angles_X -159.7524 7.2208261 

  }

  ALIstring coordi("XYZ");
  std::map<OpticalObject*,ALIRmDataFromFile>::const_iterator ite;
  for( ite = affAngles.begin(); ite != affAngles.end(); ite++ ){
    ALIRmDataFromFile dff = (*ite).second;
    OpticalObject* opto = (*ite).first;
    for( size_t ii = 0; ii < 3; ii++ ) {
      int ifound =  dff.dataFilled().find( coordi[ii] );
      if( ALIUtils::debug >= 5) std::cout << ii << " dataFilled " << ifound << std::endl; 
      if( ifound == -1 ) { //angles not read from file are taken as the original value
	ALIdouble entval = opto->getEntryRMangle(coordi.substr(ii,1));
	dff.setAngle( coordi.substr(ii,1), entval );
      } 
    }
    dff.constructRm();
    opto->setGlobalRMOriginalOriginal( dff.rm() );
  }
    
  return true; // to avoid warning
  
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIstring FittedEntriesReader::substitutePointBySlash( const ALIstring& nameWithPoints ) const
{
  ALIstring nameWithSlash = nameWithPoints;

  size_t siz = nameWithPoints.length();

  for( size_t ii = 0; ii < siz; ii++ ){
    if( nameWithSlash[ii] == '.' ) nameWithSlash[ii] = '/'; 
  }
  nameWithSlash = "s/" + nameWithSlash;
  if( ALIUtils::debug >= 5) std::cout << " substitutePointBySlash " << nameWithSlash << " " << std::endl;


  return nameWithSlash;
}
