// COCOA class implementation file
// Id:  Measurement.cc
// CAT: Model
// ---------------------------------------------------------------------------
// History: v1.0 
// Authors:
//   Pedro Arce

#include "Alignment/CocoaModel/interface/Model.h"

#include <algorithm>
#include <iomanip> 
#include <iostream>
#include <iterator>
//#include <algo.h>
#include <cstdlib>

#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/ParameterMgr.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"


ALIdouble Measurement::cameraScaleFactor = 1.;
ALIstring Measurement::theMeasurementsFileName = "";
ALIstring Measurement::theCurrentDate = "99/99/99";
ALIstring Measurement::theCurrentTime = "99:99";

ALIbool Measurement::only1 = 0;
ALIstring Measurement::only1Date = "";
ALIstring Measurement::only1Time = "";

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ constructor:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Measurement::Measurement( const ALIint measdim, ALIstring& type, ALIstring& name ) 
: theDim(measdim), theType(type), theName( name )
{
  //  _OptOnames = new ALIstring[theDim];
  theValue = new ALIdouble[theDim];
  theSigma = new ALIdouble[theDim];
  theValueType = new ALIstring[theDim];

  theValueSimulated = new ALIdouble[theDim];
  theValueSimulated_orig = new ALIdouble[theDim];
  theValueIsSimulated = new ALIbool[theDim];

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ construct (read from file)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::construct() 
{

  ALIFileIn& filein = ALIFileIn::getInstance( Model::SDFName() );

  //---------- Read OptOs that take part in this Measurement
  std::vector<ALIstring> wordlist;
  filein.getWordsInLine( wordlist );

  //--------- Fill the list of names of OptOs that take part in this measurement ( names only )
  buildOptONamesList( wordlist );
  
  if(ALIUtils::debug >= 3) {
    std::cout << "@@@@ Reading Measurement " << name() << " TYPE= " << type() << std::endl
	      << " MEASURED OPTO NAMES: ";
    std::ostream_iterator<ALIstring> outs(std::cout," ");
    copy(wordlist.begin(), wordlist.end(), outs);
    std::cout << std::endl;
  }


  //---------- Read the data 
  for ( unsigned int ii=0; ii<dim(); ii++){
    filein.getWordsInLine( wordlist );
    fillData( ii, wordlist );
  }

  if( !valueIsSimulated(0) ) correctValueAndSigma();

  postConstruct();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::constructFromOA( OpticalAlignMeasurementInfo&  measInfo) 
{
  //---- Build wordlist to build object name list
  std::vector<std::string> objNames = measInfo.measObjectNames_;
  std::vector<std::string>::const_iterator site;
  std::vector<ALIstring> wordlist;
  //--- Fill the list of names of OptOs that take part in this measurement ( names only )
  for( site = objNames.begin(); site != objNames.end(); site++) {
    if( site != objNames.begin() ) wordlist.push_back("&");
    wordlist.push_back(*site);
  }
  buildOptONamesList( wordlist );

  if(ALIUtils::debug >= 3) {
    std::cout << "@@@@ Reading Measurement " << name() << " TYPE= " << type() << " " << measInfo << std::endl
	      << " MEASURED OPTO NAMES: ";
    for( size_t ii = 0; ii < _OptONameList.size(); ii++ ){
      std::cout << _OptONameList[ii] << " ";
    }
    std::cout << std::endl;
  }

  //---------- No data, set to simulated_value
  for ( unsigned int ii=0; ii<dim(); ii++){
    wordlist.clear();
    wordlist.push_back( (measInfo.values_)[ii].name_ );
    char ctmp[20];
    if( measInfo.isSimulatedValue_[ii] ){
      if ( ALIUtils::debug >= 5 ) {
	std::cout << "Measurement::constructFromOA:  meas value " << ii << " "  << dim() << " = simulated_value" << std::endl;
      }
      wordlist.push_back("simulated_value");
    } else { 
      if ( ALIUtils::debug >= 5 ) {
	std::cout << "Measurement::constructFromOA:  meas value " << ii << " "  << dim() << " = " << measInfo.values_.size() << std::endl;
      }
      ALIdouble val = (measInfo.values_)[ii].value_ / valueDimensionFactor(); //in XML  values are without dimensions, so neutralize multiplying by valueDimensionFactor() in fillData
      gcvt( val, 10, ctmp );
      wordlist.push_back( ctmp );
    }
    ALIdouble err = (measInfo.values_)[ii].error_ / sigmaDimensionFactor(); //in XML  values are without dimensions, so neutralize multiplying by valueDimensionFactor() in fillData
    gcvt( err, 10, ctmp );
    wordlist.push_back( ctmp );
    std::cout << " sigma " << err << " = " << ctmp << " " << (measInfo.values_)[ii].error_ << std::endl;
    //-    wordlist.push_back( "simulated_value" );
    //-   wordlist.push_back( "1." );
      if ( ALIUtils::debug >= 5 ) ALIUtils::dumpVS(wordlist, " Measurement: calling fillData ");
    //-    std::cout << " MEAS INFO " << measInfo << std::endl;
    //-   std::cout << ii << " MEAS INFO PARAM " <<  (measInfo.values_)[ii] << std::endl;
    //- std::cout << ii << " MEAS INFO PARAM VALUE " <<  (measInfo.values_)[ii].value_ << std::endl;
    fillData( ii, wordlist );
  }

  postConstruct();

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::postConstruct()
{
  //---------- Set name as name of last OptO 
  setName();

  //---------- Transform for each Measurement the Measured OptO names to Measured OptO pointers
  buildOptOList();

  //---------- Build list of Entries that affect a Measurement 
  buildAffectingEntryList();
  
  //---------- add this measurement to the global list of measurements
  Model::addMeasurementToList( this );

  if ( ALIUtils::debug >= 10 ) {
    std::cout << Model::MeasurementList().size() << std::endl;
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fills a list of names of OpticalObjects that take part in this measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::buildOptONamesList( const std::vector<ALIstring>& wl ) 
{

  int NPairs = (wl.size()+1)/2;   // Number of OptO names ( pair of name and '&' )

  //--------- Fill list with names 
  for ( int ii=0; ii<NPairs; ii++ ) {
    _OptONameList.push_back( wl[ii*2] );
    // Check for separating '&'
    if (ii != NPairs-1 && wl[2*ii+1] != ALIstring("&") ) {
      //      ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
      std::cerr << "!!! Measured Optical Objects should be separated by '&', not by" 
		<< wl[2*ii+1] << std::endl; 
      exit(2);
    }
  }
 
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fill the data of measurement coordinate 'coor' with values in 'wordlist'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::fillData( ALIuint coor, const std::vector<ALIstring>& wordlist) 
{
  if ( ALIUtils::debug >= 3 ) {
    std::cout << "@@ Filling coordinate " << coor << std::endl ;
    //-   ostream_iterator<ALIstring> outs(std::cout," ");
    //-  copy(wordlist.begin(), wordlist.end(), outs);
  }
  
  ParameterMgr* parmgr = ParameterMgr::getInstance();

  //---------- Check that there are 3 attributes: name, value, error
  if( wordlist.size() != 3 ) {
    //    ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
    std::cerr << " Incorrect format for Measurement value:" << std::endl; 
    std::ostream_iterator<ALIstring> outs(std::cout," ");
    copy(wordlist.begin(), wordlist.end(), outs);
    std::cout << std::endl << "There should be three words: name value sigma " << std::endl;
    exit(2);
  }

  //---------- check coor value
  if (coor >= theDim ) {
    // ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
    std::cerr << "Trying to fill Measurement coordinate No "
	 << coor << " but the dimension is " << theDim << std::endl; 
    exit(2);
  }

  //---------- set data members 
  //----- Set valueType
  theValueType[coor] = wordlist[0];

  //----- Set value (translate it if a PARAMETER is used)
  ALIdouble val = 0.;
  theValueIsSimulated[coor] = 0;
  if( !ALIUtils::IsNumber(wordlist[1]) ) {
    if ( parmgr->getParameterValue( wordlist[1], val ) == 0 ) {
      if( wordlist[1] == ALIstring("simulated_value") ) {
	theValueIsSimulated[coor] = 1;
      } else {
	//	ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
	std::cerr << "!!! parameter for value not found: " << wordlist[1].c_str() << std::endl;
	exit(2);
      }
    }
    //d val *= valueDimensionFactor();
  } else {
    //d val = DimensionMgr()::getInstance()->extractValue( wordlist[1], ValueDimensionFactor() );
    val = atof( wordlist[1].c_str() ); 
  }
  val *= valueDimensionFactor();
  if( ALIUtils::debug >= 3 ) std::cout << "Meas VALUE= " << val << " (ValueDimensionFactor= " << valueDimensionFactor() <<std::endl;

  //----- Set sigma (translate it if a PARAMETER is used)
  ALIdouble sig = 0.;
  if( !ALIUtils::IsNumber(wordlist[2]) ) {
    if ( parmgr->getParameterValue( wordlist[2], sig ) == 0 ) {
      // ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
      std::cerr << "!!! parameter for sigma not found: " << wordlist[2].c_str() << std::endl;
      exit(2);
    }
    //d sig *= sigmaDimensionFactor();
  } else {
    //    sig = DimensionMgr()::getInstance()->extractValue( wordlist[2], ValueDimensionFactor() );
    sig = atof( wordlist[2].c_str() );  
  }
  sig *= sigmaDimensionFactor();
  if( ALIUtils::debug >= 3) std::cout << "SIGMA= " << sig << " (SigmaDimensionFactor= " << sigmaDimensionFactor() <<std::endl;
  
  //----- set theValue & theSigma
  theValue[coor] = val;
  theSigma[coor] = sig;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::fillData( ALIuint coor, OpticalAlignParam* oaParam) 
{
  if ( ALIUtils::debug >= 3 ) {
    std::cout << "@@ Filling coordinate " << coor << std::endl ;
  }
  
  //  ParameterMgr* parmgr = ParameterMgr::getInstance();

  //---------- check coor value
  if (coor >= theDim ) {
    std::cerr << "Trying to fill Measurement coordinate No "
	 << coor << " but the dimension is " << theDim << std::endl; 
    exit(2);
  }

  //---------- set data members 
  //----- Set value (translate it if a PARAMETER is used)
  ALIdouble val = 0.;
  theValueIsSimulated[coor] = 0;
  val = oaParam->value();
  val *= valueDimensionFactor();
  theValue[coor] = val;
  if( ALIUtils::debug >= 3 ) std::cout << "Meas VALUE= " << val << " (ValueDimensionFactor= " << valueDimensionFactor() <<std::endl;

  ALIbool sigmaFF = GlobalOptionMgr::getInstance()->GlobalOptions()["measurementErrorFromFile"];
  if( sigmaFF ) {
    //----- Set sigma (translate it if a PARAMETER is used)
    ALIdouble sig = 0.;
    sig = oaParam->sigma(); // it is in mm always
    sig *= sigmaDimensionFactor();
    theSigma[coor] = sig;
    if( ALIUtils::debug >= 3) std::cout << "SIGMA= " << sig << " (SigmaDimensionFactor= " << sigmaDimensionFactor() <<std::endl;
   
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Once the complete list of OptOs is read, convert OptO names to pointers
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::buildOptOList()
{
  //-  if ( ALIUtils::debug >= 3 ) std::cout << std::endl << " MEASUREMENT: " << " " << this->name() << std::endl; 
  ALIstring twopoints(".."); // .. goes one level up in the tree fo OptOs

//---------------------------------------- Loop OptONameList
  std::vector<ALIstring>::iterator vsite;
  for (vsite = _OptONameList.begin();
       vsite != _OptONameList.end(); vsite++) {
//----------------------------------- Count how many '..' there are in the name
    ALIuint ii = 0;
    //    ALIuint slen = (*vsite).length();
    ALIuint Ntwopoints = 0;    //---- No '..' in ALIstring
    for(;;) {
      int i2p = (*vsite).find_first_of( twopoints, 3*ii ); // if it is ., it also finds it!!!
      if ( i2p < 0 ) break;
      if ( i2p != ALIint(3*ii)) {
	std::cerr << i2p << "!!! Bad position of '..' in reference ALIstring: " 
	     << (*vsite).c_str() << std::endl; 
	exit(2);
      } else {
	Ntwopoints++;
	if ( ALIUtils::debug >=9 ) std::cout << "N2p" << Ntwopoints;
      }
      ii++;   
    }     
    //------ Substitute '..' by reference (the last OptO in list)
    if (Ntwopoints != 0) {
      Substitute2p( (*vsite), *(_OptONameList.end()-1), Ntwopoints);
    }
    //----- Get OpticalObject* that correspond to ALIstring and fill list
    ALIstring referenceOptO = (*vsite);
    //--- a ':' is used in OptOs that have several possible behavious
    ALIint colon = referenceOptO.find(':');
    if ( colon != -1 ) {
      if (ALIUtils::debug >=99) { 
	std::cout << "colon in reference OptO name " << colon << 
	  referenceOptO.c_str() << std::endl;
      }
      referenceOptO = referenceOptO.substr( 0, colon );
    }
    OpticalObject* OptOitem = Model::getOptOByName( referenceOptO );
    if ( ALIUtils::debug >= 3 ) std::cout << "Measurement::buildOptOList: OptO in Measurement: " << OptOitem->name() << std::endl;
    if ( OptOitem != (OpticalObject*)0 ) {
      _OptOList.push_back( OptOitem);
    } else {
      std::cerr << "!!! Error in Measurement: can't find Optical Object " <<
	(*vsite).c_str() << std::endl;
      exit(2);
    }      
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Build the list of all entries of every OptO that take part in this 
//@@ Measurement and also the list of all entries of their OptO ancestors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::buildAffectingEntryList(){
  
  //---------- Loop OptO MeasuredList
  std::vector< OpticalObject* >::const_iterator vocite;
  for (vocite = _OptOList.begin();
       vocite != _OptOList.end(); vocite++) {
      addAffectingEntriesFromOptO( *vocite );
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get the list of all entries of this OptO that take part in this Measurement
//@@ and of their OptO ancestors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::addAffectingEntriesFromOptO( const OpticalObject* optoP )
{
  if(ALIUtils::debug >= 3)  std::cout << "Measurement::addAffectingEntriesFromOptO: OptO taking part in Measurement: " << optoP->name() << std::endl;
      //---------- Loop entries in this OptO           
  std::vector< Entry* >::const_iterator vecite;
  std::vector< Entry* >::const_iterator fvecite;
  for (vecite = optoP->CoordinateEntryList().begin();
      vecite != optoP->CoordinateEntryList().end(); vecite++) {
    //T     if( find( theAffectingEntryList.begin(), theAffectingEntryList.end(), (*vecite) ) == theAffectingEntryList.end() ){
    //t      theAffectingEntryList.push_back(*vecite);
    //T    }
    fvecite = find( theAffectingEntryList.begin(), theAffectingEntryList.end(), (*vecite) );
    if (fvecite == theAffectingEntryList.end() ){
      theAffectingEntryList.push_back(*vecite);
      if(ALIUtils::debug >= 4)  std::cout << "Entry that may affect Measurement: " << (*vecite)->name() << std::endl;
    }
  }
  for (vecite = optoP->ExtraEntryList().begin();
      vecite != optoP->ExtraEntryList().end(); vecite++) {
    fvecite = find( theAffectingEntryList.begin(), theAffectingEntryList.end(), (*vecite) );
    if (fvecite == theAffectingEntryList.end() ){
      theAffectingEntryList.push_back(*vecite);
      if(ALIUtils::debug >= 4)  std::cout << "Entry that may affect Measurement: " << (*vecite)->name() << std::endl;
    }
  }
  if(optoP->parent() != 0) {
    addAffectingEntriesFromOptO( optoP->parent() );
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Substitute '..' in 'ref' by name of parent OptO ('firstref')
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::Substitute2p( ALIstring& ref, const ALIstring& firstref, int Ntwopoints) 
{
  // '/' sets hierarchy of OptOs
  ALIstring slash("/");
  
  int pos1st = firstref.length();
  // Go back an '/' in firstref for each '..' in ref
  for (int ii=0; ii < Ntwopoints; ii++) {
      pos1st = firstref.find_last_of( slash, pos1st-1);
      if ( ALIUtils::debug >=9 ) std::cout << "pos1st=" << pos1st;
  }

  if ( ALIUtils::debug >=9 ) std::cout << "before change ref: " << ref << " 1ref " << firstref << std::endl;
  // Substitute name
  ref.replace( 0, (Ntwopoints*3)-1, firstref, 0, pos1st);
  if ( ALIUtils::debug >=9 ) std::cout << "after change ref: " << ref << " 1ref " << firstref << std::endl;

}

 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::printStartCalculateSimulatedValue( const Measurement* meas) 
{
  std::cout << std::endl << "@@@ Start calculation of simulated value of " << meas->type() << " Measurement " << meas->name() << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate the simulated value of the Measurement propagating the LightRay when all the entries have their original values
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::calculateOriginalSimulatedValue() 
{
  //----------  Calculate the simulated value of the Measurement
  calculateSimulatedValue( 1 );

#ifdef COCOA_VIS
  if( ALIUtils::getFirstTime() ) {
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if(gomgr->GlobalOptions()["VisWriteVRML"] > 1) {
      ALIVRMLMgr::getInstance().newLightRay();
    }
    /*-    if(Model::GlobalOptions()["VisWriteIguana"] > 1) {
           IgCocoaFileMgr::getInstance().newLightPath( theName );
	   } */
  }
#endif

  //---------- Set original simulated values to it
  //-  if(ALIUtils::debug >= 5) std::cout << "MEAS DIMENSION" << dim() << std::endl;
  for ( ALIuint ii = 0; ii < dim(); ii++) {
    setValueSimulated_orig( ii, valueSimulated(ii) );
    if ( ALIUtils::debug >= 4 ) std::cout << "SETsimuvalOriginal" << valueSimulated(ii) << std::endl;
    //----- If Measurement has as value 'simulated_value', set the value to the simulated one
    if( valueIsSimulated(ii) == 1 ){
      setValue( ii, valueSimulated(ii) );
      //- std::cout << ii << " setting value as simulated " <<  valueSimulated(ii) << " " << value(ii) << this << std::endl;
    }
  }
 
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::DumpBadOrderOptOs() 
{
  std::cerr << " Detector can not make measurement with these optical objects " << std::endl;
  if (ALIUtils::debug >= 1) {
    //  std::vector<OpticalObject*>::iterator voite;
    //      for ( voite = _OptOList.begin(); 
    //     voite != _OptOList.end(); voite++) {
    //    std::cout << (*voite)->type() << " : " << (*voite)->name() << std::endl;
    // }
    std::vector<ALIstring>::const_iterator vsite;
    for ( vsite = OptONameList().begin(); 
	  vsite != OptONameList().end(); vsite++) {
      std::cerr << (*vsite) << " : " ;
    }
    std::cerr << std::endl;
  }
  exit(2);

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::vector<ALIdouble> Measurement::DerivativeRespectEntry( Entry* entry )
{
  //---------- std::vector of derivatives to return
  std::vector<ALIdouble> deriv;
  ALIdouble sumderiv;
  
  //---------- displacement to start with 
  ALIdouble displacement = entry->startingDisplacement();
  //----- all angles are in radians, so, if displace is not, rescale it before making the displacement 
  //-  displacement *= entry->SigmaDimensionFactor();
  if( ALIUtils::debug >= 3) std::cout << std::endl << "%%% Derivative w.r.t. entry " << entry->name() << ": displacement = " << displacement << std::endl;
  
  ALIint count_itera = 0; 
  
  //---------- Loop decreasing the displacement a factor 2, until the precision set is reached
  do {
    count_itera++;
    entry->displace( displacement );
    
    if ( ALIUtils::debug >= 5) std::cout << "Get simulated value for displacement " << displacement << std::endl;
    calculateSimulatedValue( 0 ); 
    
    //---------- Get sum of derivatives
    sumderiv = 0;   
    for ( ALIuint ii = 0; ii < theDim; ii++) {
      sumderiv += fabs( theValueSimulated[ii] - theValueSimulated_orig[ii] );
      if( ALIUtils::debug >= 4 ) {
	std::cout << "iteration " << count_itera << " COOR " << ii 
	     << " difference =" << ( theValueSimulated[ii] - theValueSimulated_orig[ii] )
	  //-		  << "  " << theValueSimulated[ii] << "  " << theValueSimulated_orig[ii] 
	     << " derivative = " << (theValueSimulated[ii] - theValueSimulated_orig[ii]) /displacement << " disp " << displacement 
	     << " sum derivatives = " << sumderiv << std::endl;
      }
      if( ALIUtils::debug >= 5 ) {
        std::cout << " new simu value= " << theValueSimulated[ii] << " orig simu value " << theValueSimulated_orig[ii] << std::endl;
       }
    }
    if (count_itera >= 100) {
      std::cerr << "EXITING: too many iterations in derivative, displacement is " <<
	displacement << " sum of derivatives is " << sumderiv << std::endl;
      exit(3);   
    }
    displacement /= 2.; 
    //-    std::cout << "sumderiv " << sumderiv << " maximu " <<  Fit::maximum_deviation_derivative << std::endl;
  }while( sumderiv > ALIUtils::getMaximumDeviationDerivative() );
  displacement *= 2; 
  
  //---------- Enough precision reached: pass result
  for ( ALIuint ii = 0; ii < theDim; ii++) {
    deriv.push_back( ( theValueSimulated[ii] - theValueSimulated_orig[ii] ) / displacement );
    //----- change it to entry sigma dimensions
    //     deriv[ii] /= entry->SigmaDimensionFactor();
    if( ALIUtils::debug >= 1) std::cout << name() << ": " <<  entry->OptOCurrent()->name() << " " <<  entry->name() << " " << ii << "### DERIVATIVE: " << deriv[ii] <<  std::endl;
  }
  //-  if(ALIUtils::debug >= 5) std::cout << "END derivative: " << deriv << "disp" << displacement << std::endl;
  
  //--------------------- Reset _centreGlob and _rmGlob of OptO entry belongs to (and component OptOs)
  entry->OptOCurrent()->resetGlobalCoordinates();
  
  return deriv;
 
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ destructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Measurement::~Measurement() 
{
  //  delete[] _name; 
  delete[] theValue;
  delete[] theSigma;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ get the ':X' that determines how the behaviour of the OptO w.r.t. this Measurement 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIstring Measurement::getMeasuringBehaviour( const std::vector< OpticalObject* >::const_iterator vocite ){ 
  std::vector<ALIstring>::const_iterator vscite = _OptONameList.begin() +
    ( vocite - _OptOList.begin() ); // point to corresponding name of this OptO
  ALIint colon = (*vscite).find(':');
  ALIstring behav;
  if(colon != -1 ) {
    behav = (*vscite).substr(colon+1,(*vscite).size());
  } else {
    behav = " ";
  }
  return behav;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get the previous OptOs in the list of OptO that take part in this measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const OpticalObject* Measurement::getPreviousOptO( const OpticalObject* Popto ) const 
{
  //--------- Loop OptOs that take part in this measurement
  std::vector<OpticalObject*>::const_iterator vocite;
  for( vocite = _OptOList.begin(); vocite != _OptOList.end(); vocite++ ){
    if( *vocite == Popto ) {
      if( vocite == _OptOList.begin() ) {
	std::cerr << " ERROR in  getPreviousOptO of measurement " << name() << std::endl;
	std::cerr << " OptO " << Popto->name() << " is the first one " << std::endl;
        exit(1);
      } else {
        return *(vocite-1);
      }
    }
  }
  
  std::cerr << " ERROR in  getPreviousOptO of measurement " << name() << std::endl;
  std::cerr << " OptO " << Popto->name() << " not found " << std::endl;
  exit(1);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::setCurrentDate( const std::vector<ALIstring>& wl )
{

  if( wl.size() != 3 ){
    std::cerr << "!!!EXITING: reading DATE of measurements set: it must have three words, it is though " << std::endl;
    ALIUtils::dumpVS(wl, " ");
   exit(1);
  } else if(wl[0] != "DATE:" ){ 
    std::cerr << "!!!EXITING: reading DATE of measurements set: first word must be 'DATE:', it is though " << std::endl;
    ALIUtils::dumpVS( wl, " ");
    exit(1);
  } else {
    theCurrentDate = wl[1];
    theCurrentTime = wl[2];
 }
} 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::copyMeas( Measurement* meas, const std::string& subsstr1, const std::string& subsstr2 )
{
  theDim = meas->dim();
  theType = meas->type();
  theName =  ALIUtils::changeName( meas->name(), subsstr1, subsstr2);

   //  _OptOnames = new ALIstring[theDim];
  theValueSimulated = new ALIdouble[theDim];
  theValueSimulated_orig = new ALIdouble[theDim];
  theValueIsSimulated = new ALIbool[theDim];
  theValue = const_cast<ALIdouble*>(meas->value());
  theSigma = const_cast<ALIdouble*>(meas->sigma());

  unsigned int ii;
  for(ii = 0; ii < theDim; ii++) {
    theValueSimulated[ii] = meas->valueSimulated( ii );
    theValueSimulated_orig[ii] = meas->valueSimulated_orig( ii );
    theValueIsSimulated[ii] = meas->valueIsSimulated( ii );
  }

  //--------- Fill the list of names of OptOs that take part in this measurement ( names only )

  std::vector<std::string> wordlist;
  auto &optolist = meas->OptOList();
  ALIuint nOptos = optolist.size();
  for ( ALIuint ii = 0; ii < nOptos; ii++ ) {
    wordlist.push_back( ALIUtils::changeName( optolist[ii]->longName(), subsstr1, subsstr2) );
    std::cout << " copymeas " << ALIUtils::changeName( optolist[ii]->longName(), subsstr1, subsstr2) << std::endl;
    if( ii != nOptos -1 ) wordlist.push_back("&");
  }

  buildOptONamesList( wordlist );
  
  if(ALIUtils::debug >= 3) {
    std::cout << "@@@@ Reading Measurement " << name() << " TYPE= " << type() << std::endl
	      << " MEASURED OPTO NAMES: ";
    std::ostream_iterator<ALIstring> outs(std::cout," ");
    copy(wordlist.begin(), wordlist.end(), outs);
    std::cout << std::endl;
  }


  postConstruct();

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Measurement::setName()
{
  // name already set by passing one argument with sensor type
  if( theName != "" ) return;
  if( _OptONameList.size() == 0) {
    std::cerr << " !!! Error in your code, you cannot ask for the name of the Measurement before the OptONameList is build " << std::endl;
    exit(9);
  }
  std::vector<ALIstring>::iterator vsite = (_OptONameList.end()-1);
  theName = type() + ":" + (*vsite);
}
