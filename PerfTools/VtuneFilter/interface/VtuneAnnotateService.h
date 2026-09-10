#pragma once
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include <vector>
#include <string>
#include <algorithm>
#include "ittnotify.h"

namespace edm {
  class VtuneAnnotateService {
  public:
    VtuneAnnotateService(const ParameterSet&, ActivityRegistry&);
    
    void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
    void postModuleEvent(StreamContext const&, ModuleCallingContext const&);

  private:
    std::vector<std::string> targetModules_;
    bool isTargetModule(std::string const& label) const;
    __itt_domain* ittDomain_; 
  };
}
