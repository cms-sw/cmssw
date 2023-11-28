#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"

std::unique_ptr<edm::ParameterSet> edm::getPSetFromConfig(const std::string& config) {
  return PyBind11ProcessDesc(config, false).parameterSet();
}

//its really the stuff in MakePythonParameterSets that should be in the different namespace
//I'll do that if this setup is ok
std::unique_ptr<edm::ParameterSet> edm::readConfig(std::string const& config, const std::vector<std::string>& args) {
  return edm::cmspybind11::readConfig(config, args);
}

std::unique_ptr<edm::ParameterSet> edm::readConfig(std::string const& config) {
  return edm::cmspybind11::readConfig(config);
}

void edm::makeParameterSets(std::string const& configtext, std::unique_ptr<ParameterSet>& main) {
  edm::cmspybind11::makeParameterSets(configtext, main);
}

std::unique_ptr<edm::ParameterSet> edm::readPSetsFrom(std::string const& fileOrString) {
  return edm::cmspybind11::readPSetsFrom(fileOrString);
}
