
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  ProcessDesc::ProcessDesc(boost::shared_ptr<ParameterSet> pset) :
      pset_(pset), services_(pset_->popVParameterSet(std::string("services")).release()) {
  }

  ProcessDesc::ProcessDesc(std::string const& config) :
      pset_(new ParameterSet),
      services_(new std::vector<ParameterSet>()) {
    throw Exception(errors::Configuration,"Old config strings no longer accepted");
  }

  ProcessDesc::~ProcessDesc() {
  }

  boost::shared_ptr<ParameterSet>
  ProcessDesc::getProcessPSet() const {
    return pset_;
  }

  boost::shared_ptr<std::vector<ParameterSet> >
  ProcessDesc::getServicesPSets() const {
    return services_;
  }

  void ProcessDesc::addService(ParameterSet& pset) {
    // The standard services should be initialized first.
    services_->insert(services_->begin(), pset);
  }

  void ProcessDesc::addService(std::string const& service) {
    ParameterSet newpset;
    newpset.addParameter<std::string>("@service_type", service);
    addService(newpset);
  }

  void ProcessDesc::addDefaultService(std::string const& service) {
    typedef std::vector<ParameterSet>::iterator Iter;
    for(Iter it = services_->begin(), itEnd = services_->end(); it != itEnd; ++it) {
      std::string name = it->getParameter<std::string>("@service_type");
      if (name == service) {
        // Use the configured service.  Don't add a default.
        // However, the service needs to be moved to the front because it is a standard service.
        ParameterSet pset = *it;
        services_->erase(it);
        addService(pset);
        return;
      }
    }
    addService(service);
  }

  void ProcessDesc::addForcedService(std::string const& service) {
    typedef std::vector<ParameterSet>::iterator Iter;
    for(Iter it = services_->begin(), itEnd = services_->end(); it != itEnd; ++it) {
      std::string name = it->getParameter<std::string>("@service_type");
      if (name == service) {
        // Remove the configured service before adding the default.
        services_->erase(it);
        break;
      }
    }
    addService(service);
  }

  void ProcessDesc::addServices(std::vector<std::string> const& defaultServices,
                                std::vector<std::string> const& forcedServices) {
    // Add the default services to services_.
    for(std::vector<std::string>::const_iterator i = defaultServices.begin(), iEnd = defaultServices.end();
         i != iEnd; ++i) {
      addDefaultService(*i);
    }
    // Add the forced services to services_.
    for(std::vector<std::string>::const_iterator i = forcedServices.begin(), iEnd = forcedServices.end();
         i != iEnd; ++i) {
      addForcedService(*i);
    }
  }

  std::string ProcessDesc::dump() const {
    std::string out = pset_->dump();
    for (std::vector<ParameterSet>::const_iterator it = services_->begin(), itEnd = services_->end(); it != itEnd; ++it) {
      out += it->dump();
    }
    return out;
  }
} // namespace edm
