//   COCOA class header file
//Id:  NtupleManager.h
//CAT: Analysis
//
//   History: v1.0 
//   Luca Scodellaro

#ifndef _NtupleManager_HH
#define _NtupleManager_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaFit/interface/MatrixMeschach.h"
#include "Alignment/CocoaFit/interface/NtupleObjects.h"
#include "CLHEP/Vector/Rotation.h"

class TFile;
class TTree;
class TClonesArray;

class NtupleManager
{

public:
  //---------- Constructors / Destructor
  NtupleManager(){ };
  ~NtupleManager(){ };
  static NtupleManager* getInstance();  
  void BookNtuple();
  void InitNtuple();
  void FillNtupleTree();
  void WriteNtuple();
  void FillChi2();
  void FillFitParameters(MatrixMeschach* AtWAMatrix);
  void FillOptObjects(MatrixMeschach* AtWAMatrix);
  void FillMeasurements();


private:
  static NtupleManager* instance;

  void GetGlobalAngles(const CLHEP::HepRotation& rmGlob, double *theta);
 
  TFile *theRootFile;

  TTree *CocoaTree;
/*   TTree *FitParametersTree; */
/*   TTree *MeasurementsTree; */

  TClonesArray* CloneFitParam;              FitParam              *FitParamA;
  TClonesArray* CloneOptObject;             OptObject             *OptObjectA;
  TClonesArray* CloneSensor2DMeas;          Sensor2DMeas          *Sensor2DMeasA;
  TClonesArray* CloneDistancemeterMeas;     DistancemeterMeas     *DistancemeterMeasA;
  TClonesArray* CloneDistancemeter1DimMeas; Distancemeter1DimMeas *Distancemeter1DimMeasA;
  TClonesArray* CloneTiltmeterMeas;         TiltmeterMeas         *TiltmeterMeasA;
  TClonesArray* CloneCopsMeas;              CopsMeas              *CopsMeasA;

/*   bool BookFitParameters; */
/*   bool BookMeasurements; */

  double Chi2Measurements, Chi2CalibratedParameters;
  int NDegreesOfFreedom;
  int NFitParameters;
  int NOptObjects;
  int NSensor2D;
  int NDistancemeter;
  int NDistancemeter1Dim;
  int NTiltmeter;
  int NCops;
};

#endif

