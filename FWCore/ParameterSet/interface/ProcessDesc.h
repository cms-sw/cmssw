#ifndef FWCore_ParameterSet_ProcessDesc_h
#define FWCore_ParameterSet_ProcessDesc_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  class ParameterSet;

  class ProcessDesc {

  public:
    explicit ProcessDesc(std::shared_ptr<ParameterSet> pset);
    explicit ProcessDesc(std::unique_ptr<ParameterSet> pset);

    /// construct from the configuration language string
    explicit ProcessDesc(std::string const& config);

    ~ProcessDesc();

    /// get the parameter set
    std::shared_ptr<ParameterSet const> getProcessPSet() const {return get_underlying_safe(pset_);}
    std::shared_ptr<ParameterSet>& getProcessPSet() {return get_underlying_safe(pset_);}

    /// get the descriptions of the services
    auto const& getServicesPSets() const {return services_;}
    auto& getServicesPSets() {return services_;}

    void addService(ParameterSet& pset);
    /// add a service as an empty pset
    void addService(std::string const& service);
    /// add a service if it's not already there
    void addDefaultService(std::string const& service);
    /// add a service and replace it if it's already there
    void addForcedService(std::string const& service);
    /// add some default services and forced services
    void addServices(std::vector<std::string> const& defaultServices,
                     std::vector<std::string> const& forcedServices = std::vector<std::string>());

    std::string dump() const;
  private:
    edm::propagate_const<std::shared_ptr<ParameterSet>> pset_;
    std::vector<ParameterSet> services_;
  };
}

#endif
