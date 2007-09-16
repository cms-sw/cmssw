#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

using namespace PhysicsTools::Calibration;

namespace { // anonymous

#ifdef STD_DICTIONARIES_STUFF_MISSING
std::vector<unsigned char> v1;
std::vector<unsigned char>::iterator v1i;
std::vector<unsigned char>::const_iterator v1ci;
std::vector<double> v2;
std::vector<double>::iterator v2i;
std::vector<double>::const_iterator v2ci;
std::vector<std::string> v3;
#endif

// MVAComputer
std::vector<Variable> v4;
std::vector<Histogram> v5;
std::vector<ProcCategory::BinLimits> v6;
std::vector<ProcLikelihood::SigBkg> v7;
std::pair<double, std::vector<double> > v8;
std::pair<std::vector<ProcMLP::Neuron>, bool> v9;
MVAComputerContainer::Entry v10;
std::vector<MVAComputerContainer::Entry> v11;

} // anonymous namespace
