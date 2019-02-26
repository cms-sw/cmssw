#ifndef FWCore_Framework_ParameterSetReader_h
#define FWCore_Framework_ParameterSetReader_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  std::unique_ptr<edm::ParameterSet> getPSetFromConfig(const std::string &config);

  std::unique_ptr<edm::ParameterSet> readConfig(std::string const& config, int argc, char* argv[]);

  std::unique_ptr<edm::ParameterSet> readConfig(std::string const& config);

  void makeParameterSets(std::string const& configtext,
			 std::unique_ptr<ParameterSet> & main);

  std::unique_ptr<edm::ParameterSet> readPSetsFrom(std::string const& fileOrString);


};
#endif


