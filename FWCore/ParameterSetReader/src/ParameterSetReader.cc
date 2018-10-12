#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

std::unique_ptr<edm::ParameterSet> edm::getPSetFromConfig(const std::string &config) {
  return PythonProcessDesc(config).parameterSet();
}


//its really the stuff in MakePythonParameterSets that should be in the different namespace
//I'll do that if this setup is ok
std::unique_ptr<edm::ParameterSet> edm::readConfig(std::string const& config, int argc, char* argv[]) {
  return edm::boost_python::readConfig(config,argc,argv);
}

std::unique_ptr<edm::ParameterSet> edm::readConfig(std::string const& config) {
  return edm::boost_python::readConfig(config);
}

void edm::makeParameterSets(std::string const& configtext,
			    std::unique_ptr<ParameterSet> & main) {
  edm::boost_python::makeParameterSets(configtext,main);
}

std::unique_ptr<edm::ParameterSet> edm::readPSetsFrom(std::string const& fileOrString) {
  return edm::boost_python::readPSetsFrom(fileOrString);
}
