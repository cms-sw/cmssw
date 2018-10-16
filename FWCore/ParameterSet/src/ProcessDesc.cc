#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  ProcessDesc::ProcessDesc(std::shared_ptr<ParameterSet> pset) :
      pset_(pset), services_(pset_->popVParameterSet(std::string("services"))) {
  }

  ProcessDesc::ProcessDesc(std::unique_ptr<ParameterSet> pset) :
    pset_(std::move(pset)), services_(pset_->popVParameterSet(std::string("services"))) {
  }

  ProcessDesc::ProcessDesc(std::string const&) :
      pset_(new ParameterSet),
      services_{} {
    throw Exception(errors::Configuration,"Old config strings no longer accepted");
  }

  ProcessDesc::~ProcessDesc() {
  }

  void ProcessDesc::addService(ParameterSet& pset) {
    // The standard services should be initialized first.
    services_.insert(services_.begin(), pset);
  }

  void ProcessDesc::addService(std::string const& service) {
    ParameterSet newpset;
    newpset.addParameter<std::string>("@service_type", service);
    addService(newpset);
  }

  void ProcessDesc::addDefaultService(std::string const& service) {
    for(auto it = services_.begin(), itEnd = services_.end(); it != itEnd; ++it) {
      std::string name = it->getParameter<std::string>("@service_type");
      if (name == service) {
        // Use the configured service.  Don't add a default.
        // However, the service needs to be moved to the front because it is a standard service.
        ParameterSet pset = *it;
        services_.erase(it);
        addService(pset);
        return;
      }
    }
    addService(service);
  }

  void ProcessDesc::addForcedService(std::string const& service) {
    for(auto it = services_.begin(), itEnd = services_.end(); it != itEnd; ++it) {
      std::string name = it->getParameter<std::string>("@service_type");
      if (name == service) {
        // Remove the configured service before adding the default.
        services_.erase(it);
        break;
      }
    }
    addService(service);
  }

  void ProcessDesc::addServices(std::vector<std::string> const& defaultServices,
                                std::vector<std::string> const& forcedServices) {
    // Add the default services to services_.
    for(auto const& service: defaultServices) {
      addDefaultService(service);
    }
    // Add the forced services to services_.
    for(auto const& service : forcedServices) {
      addForcedService(service);
    }
  }

  std::string ProcessDesc::dump() const {
    std::string out = pset_->dump();
    for (auto const& service : services_) {
      out += service.dump();
    }
    return out;
  }
} // namespace edm
