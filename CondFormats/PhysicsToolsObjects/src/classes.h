#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

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

// MVAComputer
std::vector<Variable> v19;
std::vector<ProcCategory::BinLimits> v20;
std::vector<ProcLikelihood::SigBkg> v21;
std::pair<double, std::vector<double> > v22;
std::pair<std::vector<ProcMLP::Neuron>, bool> v23;
MVAComputerContainer::Entry v24;
std::vector<MVAComputerContainer::Entry> v25;

};
} // anonymous namespace
