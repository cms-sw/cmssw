#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

using namespace PhysicsTools::Calibration;

namespace { // anonymous
namespace {

#ifdef STD_DICTIONARIES_STUFF_MISSING
std::vector<unsigned char> v1;
std::vector<double> v4;
std::vector<std::string> v7;
#endif

// Histogram
HistogramF::Range v8;
HistogramF v9;
std::vector<HistogramF> v10;
HistogramD::Range v11;
HistogramD v12;
std::vector<HistogramD> v13;

// MVAComputer
std::vector<Variable> v14;
std::vector<ProcCategory::BinLimits> v15;
std::vector<ProcLikelihood::SigBkg> v16;
std::pair<double, std::vector<double> > v17;
std::pair<std::vector<ProcMLP::Neuron>, bool> v18;
MVAComputerContainer::Entry v19;
std::vector<MVAComputerContainer::Entry> v20;

}
} // anonymous namespace
