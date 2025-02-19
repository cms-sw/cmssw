//   COCOA class header file
//Id:  NtupleManager.h
//CAT: Analysis
//
//   History: v1.0 
//   Luca Scodellaro

#ifndef _NtupleObjects_HH
#define _NtupleObjects_HH

#include "TObject.h"
#include "TString.h"
  
class FitParam : public TObject {

 public:   
  FitParam();
  ~FitParam() {}
  double InitialValue;
  double FittedValue;
  double InitialSigma; 
  double FittedSigma; 
  TString Name;
  TString Quality;
  int OptObjectIndex;

  ClassDef(FitParam,1)
};
  
class OptObject : public TObject {

 public:   
  OptObject();
  ~OptObject() {}
  double CentreGlobal[3];
  double AnglesGlobal[3];
  double CentreLocal[3];
  double AnglesLocal[3];
  TString Name;
  TString Type;
  int  Parent;

  ClassDef(OptObject,1)
};
  
class Sensor2DMeas : public TObject {

 public:   
  Sensor2DMeas();
  ~Sensor2DMeas() {}
  double Position[2];
  double PosError[2]; 
  double SimulatedPosition[2];
  TString Name;
  int OptObjectIndex;

  ClassDef(Sensor2DMeas,1)
};
   
class DistancemeterMeas : public TObject {

 public:   
  DistancemeterMeas();
  ~DistancemeterMeas() {}
  double Distance;
  double DisError; 
  double SimulatedDistance;
  TString Name;
  int OptObjectIndex;

  ClassDef(DistancemeterMeas,1)
};
  
class Distancemeter1DimMeas : public TObject {

 public:   
  Distancemeter1DimMeas();
  ~Distancemeter1DimMeas() {}
  double Distance;
  double DisError; 
  double SimulatedDistance;
  TString Name;
  int OptObjectIndex;

  ClassDef(Distancemeter1DimMeas,1)
};
  
class TiltmeterMeas : public TObject {

 public:   
  TiltmeterMeas();
  ~TiltmeterMeas() {}
  double Angle;
  double AngError;
  double SimulatedAngle;
  TString Name;
  int OptObjectIndex;

  ClassDef(TiltmeterMeas,1)
}; 
  
class CopsMeas : public TObject {

 public:   
  CopsMeas();
  ~CopsMeas() {}
  double Position[4];
  double PosError[4]; 
  double SimulatedPosition[4];
  TString Name;
  int OptObjectIndex;

  ClassDef(CopsMeas,1)
};

#endif

