#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsPerformancePayload.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"   

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromBinnedTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTFormulaPayload.h"


namespace { // anonymous
struct dictionary {

#ifdef STD_DICTIONARIES_STUFF_MISSING
std::vector<unsigned char> v1;
std::vector<double> v4;
#endif

// Histogram
PhysicsTools::Calibration::HistogramF::Range v8;
PhysicsTools::Calibration::HistogramF v9;
PhysicsTools::Calibration::HistogramF2D v10;
std::vector<PhysicsTools::Calibration::HistogramF> v11;
PhysicsTools::Calibration::HistogramD::Range v12;
PhysicsTools::Calibration::HistogramD v13;
PhysicsTools::Calibration::HistogramD2D v14;
std::vector<PhysicsTools::Calibration::HistogramD> v15;
std::vector<PhysicsTools::Calibration::HistogramF2D> v16;
std::vector<PhysicsTools::Calibration::HistogramD2D> v17;
PhysicsTools::Calibration::VHistogramD2D v18;
PhysicsTools::Calibration::HistogramD3D v30;
PhysicsTools::Calibration::HistogramF3D v31;
std::vector<PhysicsTools::Calibration::HistogramF3D> v32;
std::vector<PhysicsTools::Calibration::HistogramD3D> v33;


// MVAComputer
std::vector<PhysicsTools::Calibration::Variable> v19;
std::vector<PhysicsTools::Calibration::ProcCategory::BinLimits> v20;
std::vector<PhysicsTools::Calibration::ProcLikelihood::SigBkg> v21;
std::pair<std::vector<PhysicsTools::Calibration::ProcMLP::Neuron>, bool> v23;
PhysicsTools::Calibration::MVAComputerContainer::Entry v24;
std::vector<PhysicsTools::Calibration::MVAComputerContainer::Entry> v25;
std::vector<PhysicsTools::Calibration::VarProcessor*> v26;


// Performance DB stuff
PhysicsPerformancePayload p1;          
std::vector<PerformanceResult::ResultType> r;
std::vector<BinningVariables::BinningVariablesType> b; 
PerformancePayloadFromTable c1;

//BtagPerformancePayloadFromTable c5;
//BtagPerformancePayload c6;
PerformanceWorkingPoint c7;           

// TFormula stuff
PhysicsTFormulaPayload p10;          
std::vector<PhysicsTFormulaPayload> pv10;
PerformancePayloadFromTFormula p11;
PerformancePayloadFromBinnedTFormula p12;

}; // struct dictionary
} // anonymous namespace
