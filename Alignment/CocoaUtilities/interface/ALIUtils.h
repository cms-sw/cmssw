//   COCOA class header file
//Id:  ALIUtils.h
//CAT: Model
//
//   Class with some utility function
// 
//   History: v1.0 
//   Pedro Arce

#ifndef CocoaUtils_HH
#define CocoaUtils_HH


#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include <vector>
#include <time.h>
#include <fstream> 
#include <iostream> 

class ALIUtils
{
public:
  ALIUtils(){};
  ~ALIUtils(){};

  static int IsNumber( const ALIstring& str);
  static void dump3v( const CLHEP::Hep3Vector& vec, const std::string& msg);
  static void dumprm( const CLHEP::HepRotation& rm, const std::string& msg, std::ostream& out = std::cout );

 // public static DATA MEMBERS 
  static ALIint report;
  static ALIint debug;
  static ALIdouble deg;

  static void setReportVerbosity( ALIint val ) {
    report = val;
  }
  static void setDebugVerbosity( ALIint val ) {
    debug = val;
  }
  static time_t time_now() {
    return _time_now;
  }
  static void set_time_now(time_t now) {
    _time_now = now;
  }
  //! Convert a string to an float, checking that it is really a number
  static double getFloat( const ALIstring& str );
  //! Convert a string to an integer, checking that it is really an integer
  static int getInt( const ALIstring& str );
  //! Convert a bool to an integer, checking that it is really a bool
  static bool getBool( const ALIstring& str );
  //! dumps a vector of strings with a message to outs
  static void dumpVS( const std::vector<ALIstring>& wl , const std::string& msg, std::ostream& outs = std::cout) ;
 
 //---------- Dimension factors
  static void SetLengthDimensionFactors(); 
  static void SetAngleDimensionFactors(); 
  static void SetOutputLengthDimensionFactors(); 
  static void SetOutputAngleDimensionFactors(); 
  static ALIdouble CalculateLengthDimensionFactorFromInt( ALIint ad );
  static ALIdouble CalculateAngleDimensionFactorFromInt( ALIint ad );
  static ALIdouble CalculateLengthDimensionFactorFromString( ALIstring dimstr );
  static ALIdouble CalculateAngleDimensionFactorFromString( ALIstring dimstr );

  static void dumpDimensions( std::ofstream& fout );

  static ALIdouble LengthValueDimensionFactor(){
    return _LengthValueDimensionFactor;}
  static ALIdouble LengthSigmaDimensionFactor(){
    return _LengthSigmaDimensionFactor;}
  static ALIdouble AngleValueDimensionFactor(){
    return _AngleValueDimensionFactor;}
  static ALIdouble AngleSigmaDimensionFactor(){
    return _AngleSigmaDimensionFactor;}
  static ALIdouble OutputLengthValueDimensionFactor(){
    return _OutputLengthValueDimensionFactor;}
  static ALIdouble OutputLengthSigmaDimensionFactor(){
    return _OutputLengthSigmaDimensionFactor;}
  static ALIdouble OutputAngleValueDimensionFactor(){
    return _OutputAngleValueDimensionFactor;}
  static ALIdouble OutputAngleSigmaDimensionFactor(){
    return _OutputAngleSigmaDimensionFactor;}

  static ALIdouble val0( ALIdouble val ) {
    //-std::cout << val << " val " << ( (val <= 1.E-9) ? 0. : val) << std::endl; 
//    return (abs(val) <= 1.E-9) ? 0. : val; }
    if( fabs(val) <= 1.E-9) { return 0.;
    }else { return val; }; }

  static ALIstring subQuotes( const ALIstring& str );

  static ALIdouble getDimensionValue( const ALIstring& dim, const ALIstring& dimType );

  static std::string changeName( const std::string& oldName, const std::string& subsstr1,  const std::string& subsstr2 );

  static ALIbool getFirstTime(){
    return firstTime;
  }
  static void setFirstTime( ALIbool val ){
    firstTime = val;
  }
  static ALIdouble getMaximumDeviationDerivative() {
    return maximum_deviation_derivative; }
  static void setMaximumDeviationDerivative( ALIdouble val ) {
    maximum_deviation_derivative = val; }

  static std::vector<double> getRotationAnglesFromMatrix( CLHEP::HepRotation& rmLocal, double origAngleX, double origAngleY, double origAngleZ );
  static double diff2pi( double ang1, double ang2 );
  static bool eq2ang( double ang1, double ang2 );
  static double approxTo0( double val );
  static double addPii( double val );
  static int checkMatrixEquations( double angleX, double angleY, double angleZ, CLHEP::HepRotation* rot);


 private:
  static ALIdouble _LengthValueDimensionFactor;
  static ALIdouble _LengthSigmaDimensionFactor;
  static ALIdouble _AngleValueDimensionFactor;
  static ALIdouble _AngleSigmaDimensionFactor;
  static ALIdouble _OutputLengthValueDimensionFactor;
  static ALIdouble _OutputLengthSigmaDimensionFactor;
  static ALIdouble _OutputAngleValueDimensionFactor;
  static ALIdouble _OutputAngleSigmaDimensionFactor;
  static time_t _time_now;

  static ALIbool firstTime;

  static ALIdouble maximum_deviation_derivative;

};

/*
template<class T>
ALIuint FindItemInVector( const T* item, const std::vector<T*>& item_vector )
{
 
  std::vector<T*>::const_iterator vtcite;
  ALIuint vfound = 1;
  for( vtcite = item_vector.begin(); vtcite != item_vector.end(); vtcite++) {
    if( (*vtcite) == item ) {
      
    }
  }

}
*/
//std::ostream& operator << (std::ostream& os, const CLHEP::HepRotation& c);

#endif 


