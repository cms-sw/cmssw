#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

using namespace PhysicsTools::Calibration;

namespace { // anonymous

// MVAComputer
std::vector<unsigned char> v1;
std::vector<unsigned char>::iterator v1i;
std::vector<unsigned char>::const_iterator v1ci;
std::vector<double> v2;
std::vector<double>::iterator v2i;
std::vector<double>::const_iterator v2ci;
std::vector<std::string> v3;
std::vector<Variable> v4;
std::vector<PDF> v5;
std::vector<ProcLikelihood::SigBkg> v6;
std::pair<double, std::vector<double> > v7;
std::pair<std::vector<ProcMLP::Neuron>, bool> v8;
MVAComputerContainer::Entry v9;
std::vector<MVAComputerContainer::Entry> v10;

} // anonymous namespace
