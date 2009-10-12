#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsPerformancePayload.h"


#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"   


//
// tformula stuff
//
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTFormulaPayload.h"

using namespace PhysicsTools::Calibration;

namespace { // anonymous
struct dictionary {

#ifdef STD_DICTIONARIES_STUFF_MISSING
std::vector<unsigned char> v1;
std::vector<double> v4;
std::vector<std::string> v7;
#endif

// Histogram
HistogramF::Range v8;
HistogramF v9;
HistogramF2D v10;
std::vector<HistogramF> v11;
HistogramD::Range v12;
HistogramD v13;
HistogramD2D v14;
std::vector<HistogramD> v15;
std::vector<HistogramF2D> v16;
std::vector<HistogramD2D> v17;
PhysicsTools::Calibration::VHistogramD2D v18;
HistogramD3D v30;
HistogramF3D v31;
std::vector<HistogramF3D> v32;
std::vector<HistogramD3D> v33;




// MVAComputer
std::vector<Variable> v19;
std::vector<ProcCategory::BinLimits> v20;
std::vector<ProcLikelihood::SigBkg> v21;
std::pair<double, std::vector<double> > v22;
std::pair<std::vector<ProcMLP::Neuron>, bool> v23;
MVAComputerContainer::Entry v24;
std::vector<MVAComputerContainer::Entry> v25;

// Performance DB stuff
    PhysicsPerformancePayload p1;          
    std::vector<PerformanceResult::ResultType> r;
    std::vector<BinningVariables::BinningVariablesType> b; 
    PerformancePayloadFromTable c1;
    
    // BtagPerformancePayloadFromTable c5;
    //BtagPerformancePayload c6;
    PerformanceWorkingPoint c7;           
    //
    // tformula stuff
    //
    PhysicsTFormulaPayload p10;          
    PerformancePayloadFromTFormula p11;
    std::vector<std::pair<float,float> >  p12;
    std::pair<float,float>   p13;

};
} // anonymous namespace
