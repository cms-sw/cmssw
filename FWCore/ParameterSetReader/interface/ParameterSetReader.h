#ifndef FWCore_Framework_ParameterSetReader_h
#define FWCore_Framework_ParameterSetReader_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <utility>
#include <vector>

namespace edm {

  std::unique_ptr<edm::ParameterSet> getPSetFromConfig(const std::string& config);

  // read the configuration, then set the given process options: {"numberOfThreads", "4"} is equivalent to
  //   process.options.numberOfThreads = 4
  std::unique_ptr<edm::ParameterSet> readConfig(
      std::string const& config,
      const std::vector<std::string>& args,
      std::vector<std::pair<std::string, std::string>> const& processOptions = {});

  std::unique_ptr<edm::ParameterSet> readConfig(std::string const& config);

  void makeParameterSets(std::string const& configtext,
                         std::unique_ptr<ParameterSet>& main,
                         std::vector<std::pair<std::string, std::string>> const& processOptions = {});

  std::unique_ptr<edm::ParameterSet> readPSetsFrom(std::string const& fileOrString);

}  // namespace edm
#endif
