// COCOA class header file
// Id:  MeasurementDiffEntry.h
// CAT: Model
//
// Class for measurements
//
// History: v1.0
// Authors:
//   Pedro Arce

#ifndef _MeasurementDiffEntry_HH
#define _MeasurementDiffEntry_HH

#include <vector>
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementDiffEntry : public Measurement {
public:
  MeasurementDiffEntry(const ALIint measdim, ALIstring& type, ALIstring& name) : Measurement(measdim, type, name){};
  MeasurementDiffEntry(){};
  ~MeasurementDiffEntry() override{};

  // separate OptO names and Entry names
  void buildOptONamesList(const std::vector<ALIstring>& wl) override;

  // Get simulated value (called every time a parameter is displaced)
  void calculateSimulatedValue(ALIbool firstTime) override;

private:
  ALIstring theEntryNameFirst;
  ALIstring theEntryNameSecond;
};

#endif
