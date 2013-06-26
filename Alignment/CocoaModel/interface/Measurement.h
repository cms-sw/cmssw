// COCOA class header file
// Id:  Measurement.h
// CAT: Model
//
// Class for measurements
// 
// History: v1.0.0
// v1.1.0: add measurementsFileName
// 
// Authors:
//   Pedro Arce

#ifndef _MEASUREMENT_HH
#define _MEASUREMENT_HH

#include <vector>
#include <cstdlib>

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
class OpticalObject;
class Entry;
class EntryLength;
class OpticalAlignMeasurementInfo;
class OpticalAlignParam;

class Measurement
{ 
public:
  //----- Constructors / destructor
  Measurement( const ALIint measdim, ALIstring& type, ALIstring& name );
  Measurement(){ };   
  virtual ~Measurement();
    
  // construct Measurement reading date from file
  void construct();
  void postConstruct();
  // Fill the list of names of OptOs that take part in this measurement ( names only )
  virtual void buildOptONamesList( const std::vector<ALIstring>& wl );
  // Fill the data 
  void fillData(ALIuint coor, const std::vector<ALIstring>& wl );
  void fillData( ALIuint coor, OpticalAlignParam* oaParam);

  // Convert OptOs names in OptOs pointers
  void buildOptOList();
  // Make list including every entry of every ancestor of each Measured OptO
  void buildAffectingEntryList();
  void addAffectingEntriesFromOptO( const OpticalObject* optoP );

  // Get simulated value (called every time a parameter is displaced)
  virtual void calculateSimulatedValue( ALIbool firstTime ) {};
  // Get simulated value original (called every time a parameter value is changed: after getting values from file and every non-linear fit iteration )
  void calculateOriginalSimulatedValue();

  // Dump the list of OptO names (used mainly when checking their order)
  void DumpBadOrderOptOs();

  // Calculate derivative of this Measurement with respect to a parameter of an Entry
  std::vector<ALIdouble> DerivativeRespectEntry( Entry* entry );

  // get the ':X' that determines how the behaviour of the OptO w.r.t. this Measurement
  ALIstring getMeasuringBehaviour( const std::vector< OpticalObject* >::const_iterator vocite);

 // Get the previous OptOs in the list of OptO that take part in this measurement
  const OpticalObject* getPreviousOptO( const OpticalObject* Popto ) const;
  //---------- Add any correction between the measurement data and the default format in COCOA
  virtual void correctValueAndSigma(){};

  //---------- Convert from V to rad
  virtual void setConversionFactor( const std::vector<ALIstring>& wordlist ){
    std::cerr << " Measurement::setConversionFactor should never be called " << std::endl;
    exit(1); };

  //! set the date of the current measurement
  static void setCurrentDate( const std::vector<ALIstring>& wl );

  void copyMeas( Measurement* meas, const std::string& subsstr1, const std::string& subsstr2 );

  void constructFromOA( OpticalAlignMeasurementInfo&  measInfo);

 // ACCESS DATA MEMBERS
  const ALIuint dim() const { 
    return theDim;
  }
 
  const ALIstring& type() const {
    return theType;
  }
  
  const ALIstring& name() const {
    return theName;
  }

  const ALIstring& sensorName() {
    ALIstring sensName = theName;
    ALIint colon = theName.find(':');
    theName = theName.substr(colon+1, theName.length()-colon);
    return theName;
  }
  
  //  const OpticalObject* OptOCurrent() const {
  //  return _OptOCurrent;
  // }
  
  const std::vector<ALIstring>& OptONameList() const {
    return _OptONameList;
  }
  
  const std::vector<OpticalObject*>& OptOList() const {
    return _OptOList;
  }
  
  const std::vector<Entry*>& affectingEntryList() const {
    return theAffectingEntryList;
  }
  
  const  ALIdouble valueSimulated( ALIuint ii ) const {
    return theValueSimulated[ii];
  }

  const ALIdouble valueSimulated_orig( ALIuint ii ) const {
    return theValueSimulated_orig[ii];
  }

  const ALIdouble* value() const {
    return theValue;
  } 
  const ALIdouble value( ALIuint ii ) const {
    return theValue[ii];
  } 

  const ALIdouble* sigma() const {
    return theSigma;
  } 

  const ALIdouble sigma( ALIuint ii) const {
    return theSigma[ii];
  } 

  const ALIstring valueType( ALIuint ii) const {
    return theValueType[ii];
  } 

  virtual const ALIdouble valueDimensionFactor() const{
    return ALIUtils::LengthValueDimensionFactor();
  }

  virtual const ALIdouble sigmaDimensionFactor() const{
    return ALIUtils::LengthSigmaDimensionFactor();
  }

  static ALIstring getCurrentDate(){
    return theCurrentDate;
  }
  static ALIstring getCurrentTime(){
    return theCurrentTime;
  }

  const CLHEP::Hep3Vector& getLightRayPosition( ) const{
    return theLightRayPosition; 
  }
  const CLHEP::Hep3Vector& getLightRayDirection( ) const{ 
    return theLightRayDirection; 
  }

 // SET DATA MEMBERS
  void setValue( ALIint coor, ALIdouble val) {
    theValue[coor] = val;
  }
  
  void setSigma( ALIint coor, ALIdouble val) {
    theSigma[coor] = val;
    //-    std::cout << coor << " setting sigma " << theSigma[coor] << std::endl;
  }

  void setType( ALIstring type ) {
    theType = type;
  }

  void SetDimension(ALIuint dim) {
    theDim = dim;
  }    

  void AddOptONameListItem(ALIstring optos) {
      _OptONameList.push_back( optos );
  }

  void AddOptOListItem(OpticalObject* opto) {
      _OptOList.push_back( opto );
  }

  void setValueSimulated_orig( ALIint coor, ALIdouble value) {
      theValueSimulated_orig[coor] = value;
  }

  void setValueSimulated( ALIint coor, ALIdouble value) {
      theValueSimulated[coor] = value;
  }
  virtual int xlaserLine( ALIuint ii) { std::cerr << "!!!! Measurement::xlaserLine is not returning anything " << std::endl; abort(); };
 
  //----- Set name as type plus name of last OptO 
  void setName();

  // Check is value is simulated
  bool valueIsSimulated(ALIint coor) {
      return theValueIsSimulated[coor];
  }

  virtual void setXlaserLine( ALIuint ii, int val ) { };

 static ALIdouble cameraScaleFactor;

 static ALIstring& measurementsFileName() { 
   return theMeasurementsFileName;
 }
 static void setMeasurementsFileName( const ALIstring& filename ) { 
   //-   std::cout << " setting file name " << filename << std::endl;
   theMeasurementsFileName = filename;
   //-   std::cout << " dsetting file name " << filename << std::endl;
 }

 void setLightRayPosition( const CLHEP::Hep3Vector& lightRayPosition )
   { theLightRayPosition = lightRayPosition; }
 void setLightRayDirection( const CLHEP::Hep3Vector& lightRayDirection )
   { theLightRayDirection = lightRayDirection; }

 protected:  
  // Substitute '..' by parent OptO in name 
  void Substitute2p( ALIstring& ref, const ALIstring& firstref, int NtwoPoints);
  void printStartCalculateSimulatedValue( const Measurement* meas);


 // private DATA MEMBERS
private:
  ALIuint theDim;
  ALIstring theType;
  ALIdouble* theValue;  
  ALIdouble* theSigma;
  ALIstring theName;  //name of last OptO
  ALIstring* theValueType;  //type of each measurement value (e.g. H:, TA:)

  //----- values of measurement obtained simulating the light ray through all the OptO that take part in the measurement
  ALIdouble* theValueSimulated;
  //----- values of measurement obtained simulating the light ray through all the OptO that take part in the measurement, for original values of every entry
  ALIdouble* theValueSimulated_orig;

  //-  ALIdouble* theSigmaErrorPropagation;
  //-  ALIdouble* theSigmaRegression;

  //----- Boolean to indicate if theValueSimulated_orig is set equal to the simulated values with original entries
  ALIbool* theValueIsSimulated;

  //----- List of OptOs that take part in this measurement ( names only )
  std::vector<ALIstring> _OptONameList;
  //----- List of OptOs that take part in this measurement ( pointers )
  std::vector<OpticalObject*> _OptOList;
  //----- List of OptOs Measured and their ancestors
  std::vector<Entry*> theAffectingEntryList;

  CLHEP::Hep3Vector theLightRayPosition;
  CLHEP::Hep3Vector theLightRayDirection;
  static ALIstring theMeasurementsFileName;

  static ALIstring theCurrentDate;  
  static ALIstring theCurrentTime;  
 public:
  static ALIbool only1;
  static ALIstring only1Date;
  static ALIstring only1Time;

};

#endif
