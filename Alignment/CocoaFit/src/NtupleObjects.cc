//   COCOA class implementation file
//Id:  NtupleManager.cc
//CAT: Analysis
//
//   History: v1.0  
//   Luca Scodellaro
#include "Alignment/CocoaFit/interface/NtupleObjects.h"

ClassImp(FitParam)
ClassImp(OptObject)
ClassImp(Sensor2DMeas)
ClassImp(DistancemeterMeas) 
ClassImp(Distancemeter1DimMeas)
ClassImp(TiltmeterMeas)
ClassImp(CopsMeas)  

FitParam::FitParam() {
  Name = "Null";
  Quality = "Null";
  InitialValue = -999.;
  FittedValue = -999.;
  InitialSigma = -999.; 
  FittedSigma = -999.; 
  OptObjectIndex = -999;
}

OptObject::OptObject() {
  Name = "Null";
  Type = "Null";
  Parent = -999;
  for (int i = 0; i<3; i++) { 
    CentreGlobal[i] = -999.; 
    AnglesGlobal[i] = -999.; 
    CentreLocal[i] = -999.; 
    AnglesLocal[i] = -999.; 
  }
}

Sensor2DMeas::Sensor2DMeas() {
  Name = "Null";
  OptObjectIndex = -999;
  for (int i = 0; i<2; i++) {
    Position[i] = -999.; 
    PosError[i] = -999.; 
    SimulatedPosition[i] = -999.; 
  }
}

DistancemeterMeas::DistancemeterMeas() {
  Name = "Null";
  OptObjectIndex = -999;
  Distance = -999.; 
  DisError = -999.; 
  SimulatedDistance = -999.; 
}

Distancemeter1DimMeas::Distancemeter1DimMeas() {
  Name = "Null";
  OptObjectIndex = -999;
  Distance = -999.; 
  DisError = -999.; 
  SimulatedDistance = -999.; 
}

TiltmeterMeas::TiltmeterMeas() {
  Name = "Null";
  OptObjectIndex = -999;
  Angle = -999.;
  AngError = -999.; 
  SimulatedAngle = -999.; 
}

CopsMeas::CopsMeas() {
  Name = "Null";
  OptObjectIndex = -999;
  for (int i = 0; i<4; i++) {
    Position[i] = -999.; 
    PosError[i] = -999.; 
    SimulatedPosition[i] = -999.; 
  }
}
