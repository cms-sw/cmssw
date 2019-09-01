#ifndef SherpackFetcher_h
#define SherpackFetcher_h

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cstdint>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "GeneratorInterface/SherpaInterface/interface/SherpackUtilities.h"

namespace spf {

  class SherpackFetcher {
  public:
    SherpackFetcher(edm::ParameterSet const &);
    int Fetch();
    ~SherpackFetcher();
    int CopyFile(std::string pathstring);
    const char *classname() const { return "SherpackFetcher"; }

  private:
    std::string SherpaProcess;
    std::string SherpackLocation;
    std::string SherpackChecksum;
    bool FetchSherpack;
    std::string SherpaPath;
  };

}  // namespace spf

#endif
