#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/PhysicsToolsObjects/src/headers.h"

int main() {
  testSerialization<BinningVariables>();
  //testSerialization<PerformancePayload>(); abstract
  testSerialization<PerformancePayloadFromBinnedTFormula>();
  testSerialization<PerformancePayloadFromTFormula>();
  testSerialization<PerformancePayloadFromTable>();
  testSerialization<PerformanceResult>();
  testSerialization<PerformanceWorkingPoint>();
  testSerialization<PhysicsPerformancePayload>();
  testSerialization<PhysicsTFormulaPayload>();
  testSerialization<PhysicsTGraphPayload>();
  testSerialization<PhysicsTools::Calibration::BitSet>();
  testSerialization<PhysicsTools::Calibration::Histogram2D<double, double, double>>();
  testSerialization<PhysicsTools::Calibration::Histogram2D<float, float, float>>();
  testSerialization<PhysicsTools::Calibration::Histogram3D<double, double, double, double>>();
  testSerialization<PhysicsTools::Calibration::Histogram3D<float, float, float, float>>();
  testSerialization<PhysicsTools::Calibration::Histogram<double, double>>();
  testSerialization<PhysicsTools::Calibration::Histogram<float, float>>();
  testSerialization<PhysicsTools::Calibration::MVAComputer>();
  testSerialization<PhysicsTools::Calibration::MVAComputerContainer>();
  testSerialization<PhysicsTools::Calibration::MVAComputerContainer::Entry>();
  testSerialization<PhysicsTools::Calibration::Matrix>();
  testSerialization<PhysicsTools::Calibration::ProcCategory>();
  testSerialization<PhysicsTools::Calibration::ProcClassed>();
  testSerialization<PhysicsTools::Calibration::ProcCount>();
  testSerialization<PhysicsTools::Calibration::ProcExternal>();
  testSerialization<PhysicsTools::Calibration::ProcForeach>();
  testSerialization<PhysicsTools::Calibration::ProcLikelihood>();
  //testSerialization<PhysicsTools::Calibration::ProcLikelihood::SigBkg>(); has uninitialized booleans
  testSerialization<PhysicsTools::Calibration::ProcLinear>();
  testSerialization<PhysicsTools::Calibration::ProcMLP>();
  testSerialization<PhysicsTools::Calibration::ProcMatrix>();
  testSerialization<PhysicsTools::Calibration::ProcMultiply>();
  testSerialization<PhysicsTools::Calibration::ProcNormalize>();
  testSerialization<PhysicsTools::Calibration::ProcOptional>();
  testSerialization<PhysicsTools::Calibration::ProcSort>();
  testSerialization<PhysicsTools::Calibration::ProcSplitter>();
  testSerialization<PhysicsTools::Calibration::Range<double>>();
  testSerialization<PhysicsTools::Calibration::Range<float>>();
  testSerialization<PhysicsTools::Calibration::VHistogramD2D>();
  testSerialization<PhysicsTools::Calibration::VarProcessor>();
  testSerialization<PhysicsTools::Calibration::Variable>();
  testSerialization<std::vector<BinningVariables::BinningVariablesType>>();
  testSerialization<std::vector<PerformanceResult::ResultType>>();
  testSerialization<std::vector<PhysicsTFormulaPayload>>();
  testSerialization<std::vector<PhysicsTGraphPayload>>();
  testSerialization<std::vector<PhysicsTools::Calibration::HistogramD2D>>();
  testSerialization<std::vector<PhysicsTools::Calibration::HistogramD3D>>();
  testSerialization<std::vector<PhysicsTools::Calibration::HistogramD>>();
  testSerialization<std::vector<PhysicsTools::Calibration::HistogramF2D>>();
  testSerialization<std::vector<PhysicsTools::Calibration::HistogramF3D>>();
  testSerialization<std::vector<PhysicsTools::Calibration::HistogramF>>();
  testSerialization<std::vector<PhysicsTools::Calibration::MVAComputerContainer::Entry>>();
  testSerialization<std::vector<PhysicsTools::Calibration::ProcLikelihood::SigBkg>>();
  testSerialization<std::vector<PhysicsTools::Calibration::VarProcessor*>>();
  testSerialization<std::vector<PhysicsTools::Calibration::Variable>>();

  return 0;
}
