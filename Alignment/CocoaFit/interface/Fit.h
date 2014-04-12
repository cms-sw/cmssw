//   COCOA class header file
//Id:  Fit.h
//CAT: Model
//
//   Utility class that starts reading the system description file
//              and contains the static data 
// 
//   History: v1.0 
//   Pedro Arce
#ifndef FIT_H
#define FIT_H

#define MAT_MESCHACH
#ifdef MAT_MESCHACH
#include "Alignment/CocoaFit/interface/MatrixMeschach.h"
#endif


#include <vector>


#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 
class Entry;
class OpticalObject;
#include "CLHEP/Vector/Rotation.h"
class FittedEntriesSet;
class ALIFileOut;
#include "Alignment/CocoaModel/interface/Model.h"

typedef std::pair<ALIstring, ALIstring> pss;

enum FitQuality{FQsmallDistanceToMinimum,FQbigDistanceToMinimum,FQchiSquareWorsened};

class Fit
{
public:
  //----- Constructor / Destructor
  Fit(){ };
  ~Fit(){ };

  //----- Gets the only instance of this class
  static Fit& getInstance();  

  //----- Steering method to make the fit
  static void startFit();

  //----- Dump all the entries that have been fitted (those that were 'cal' or 'unk'
  //deprecated  static void dumpFittedEntries();

  //----- Dump all the entries of all the objects 
  static void dumpFittedValues( ALIFileOut& fileout, ALIbool printErrors = 1, ALIbool printOrig = 1 );
  static void dumpFittedValuesInAllAncestorFrames( ALIFileOut& fileout, ALIbool printErrors = 1, ALIbool printOrig = 1 );
  static void dumpEntryAfterFit( ALIFileOut& fileout, const Entry* entry, double entryvalue, ALIbool printErrors = 1, ALIbool printOrig = 1 );
  static void dumpEntryCorrelations( ALIFileOut& file );
  static void printCentreInOptOFrame( const OpticalObject* opto, const OpticalObject* optoAncestor, ALIFileOut& fileout, ALIbool printErrors = 1, ALIbool printOrig = 1 );
  static void printRotationAnglesInOptOFrame( const OpticalObject* opto, const OpticalObject* optoAncestor, ALIFileOut& fileout, ALIbool printErrors = 1, ALIbool printOrig = 1 );

  //----- Dump matrices used for the fit
  static void dumpMatrices();


 //@@@@ Access static data members

  //-----  Search an Entry name in of OpticalObject 'opto_name' and return its fitpos
  static ALIint findEntryFitPosition( const ALIstring& opto_name, const ALIstring& entry_name ); 

  //----- returns propagation matrix
  //op  static ALIMatrix& propagationMatrix(){
 //op     return *thePropagationMatrix;
 //op   }


//@@@@@ private METHODS
 public:
  //------ Count how many entries are going to be fitted (have quality >=  theMinimumEntryQuality). Set for this entries the value of theFitPos
  static void setFittableEntries();
  
  static ALIbool fitNextEvent( ALIuint& nEvent );

 private:
  static void WriteVisualisationFiles();

  //----- Calculate the parameters (position, angles,...) with the chi square fit 
  //---- daFactor is the factor by which it is multiplied the matrix Da
  static FitQuality fitParameters( const double daFactor ); 

  static void redoMatrices();

  //----- Propagate the error of every Entry to every Measurement
  static void PropagateErrors();

 public:
  //----- Calculate the simulated value of each Measurement propagating the LightRay when all the entries have their original values
  static void calculateSimulatedMeasurementsWithOriginalValues();

  static ALIint noFitIterations(){ return theNoFitIterations; }

 private:
  //----- Calculate the NoLines & NoColumns and create matrices 
  static void CreateMatrices();

  //-----   Loop Measurements:
  //---  Fill Matrix A with derivatives respect to affecting entries 
  //---  Fill Matrix W, y & f with values and sigmas of measurement coordinateFill matrices  
  static void FillMatricesWithMeasurements();

  //----- Loop Measurements:
  //---    Fill Matrix A with derivatives respect to affecting entries 
  //---    Fill Matrix W, y & f with values and sigmas of measurement coordinate
  static void FillMatricesWithCalibratedParameters();

  //----- set Correlations In W Matrix
  static void setCorrelationsInWMatrix();

  //---------- set correlation between two entries of two OptOs
  static void setCorrelationFromParamFitted( const pss& entry1, const pss& entry2, ALIdouble correl );
  static void setCorrelationFromParamFitted( const ALIint fit_pos1, const ALIint fit_pos2, ALIdouble correl );

  //----- multiply matrices needed for fit
  static void multiplyMatrices();

  //----- Get Smatrix to calculate chi2
  static ALIdouble GetSChi2( ALIbool useDa );

  //----- check if the quality of the fit for this iteration is good enough
  static FitQuality getFitQuality( const ALIbool canBeGood = TRUE );
  static void evaluateFitQuality( const FitQuality fq, const double daFactor );

  //----- Correct entries with fitted values  
  static void addDaMatrixToEntries();

  //----- Substract Da of previous iteration (to try with a new Correct entries with fitted values  
  static void substractLastDisplacementToEntries( const ALIdouble factor );

  static void deleteMatrices();

  static double getEntryValue( const Entry* entry );

 public:
  static void PrintChi2( ALIdouble fit_quality, ALIbool isFirst );

 // public static DATA MEMBERS 
public:
  // maximum deviation in a Measurent when a parameter is  displaced to get derivative

  static ALIMatrix* GetAtWAMatrix(){ 
    return AtWAMatrix; }

 private:

  static void CheckIfFitPossible();
  static int CheckIfMeasIsProportionalToAnother( ALIuint measNo );
  static std::string GetMeasurementName( int meas );

// private DATA MEMBERS 
private:
  // Only instance of Fit
  static Fit* instance;

  static ALIMatrix* AMatrix;
  static ALIMatrix* AtMatrix;
  static ALIMatrix* WMatrix;
  static ALIMatrix* AtWAMatrix;
  //op  static ALIMatrix* VaMatrix;
  static ALIMatrix* DaMatrix;
  //op  static ALIMatrix* PDMatrix;
  //-  static ALIMatrix* VyMatrix;
  //op  static ALIMatrix* yMatrix;
  //op  static ALIMatrix* fMatrix;
  static ALIMatrix* yfMatrix;
  //op  static ALIMatrix* thePropagationMatrix;
  //----- The number of lines and columns of matrix A
  static ALIint _NoLinesA;
  static ALIint _NoColumnsA;

  //FOR LINK..................
private:
    //
  //-  void AddSigma( CLHEP::Hep3Vector& vori, CLHEP::Hep3Vector& vadd );
  //-  CLHEP::Hep3Vector atanVectorSigma( CLHEP::Hep3Vector& tanvs, const CLHEP::Hep3Vector& tanv );
  //-  CLHEP::Hep3Vector atanVector( CLHEP::Hep3Vector& tanv );

  //----- The minimum quality an entry must have to be inhcluded in the fit
  static ALIint theMinimumEntryQuality; 

  //----- Quality of fit in previous iteration 
  static ALIdouble thePreviousIterationFitQuality;

  //----- Minimum quality (chi square) the fit has to have to be good
  static ALIdouble theFitQualityCut;

  //----- Minimum change in quality (chi square) w.r.t previous iteration the fit has to have to be good
  static ALIdouble theRelativeFitQualityCut;

  //----- Number of fit iterations made up to a certain moment
  static ALIint theNoFitIterations;
  //----- Maximum number of fit iterations for the fit to reach good quality
  static ALIint MaxNoFitIterations;

  //----- Minimum value of the factor to multiply Da to test if an smaller chi2 can be obtained (chapter 5.2 of LeastSquareMethods)
  static ALIdouble theMinDaFactor;

 public:
  static ALIuint nEvent; 
};


#endif 


