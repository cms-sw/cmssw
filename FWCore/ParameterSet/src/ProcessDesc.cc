
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  ProcessDesc::ProcessDesc(ParameterSet& pset) :
      pset_(new ParameterSet(pset)), services_(new std::vector<ParameterSet>) {
    std::auto_ptr<ParameterSet> services = pset.popParameterSet(std::string("services"));
    std::vector<std::string> serviceNames;
    services->getParameterSetNames(serviceNames);
    for(std::vector<std::string>::const_iterator it = serviceNames.begin(), itEnd = serviceNames.end();
        it != itEnd; ++it) {
      services_->push_back(services->getUntrackedParameter<ParameterSet>(*it));
    }
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
    services_->push_back(pset);
  }

  void ProcessDesc::addService(std::string const& service) {
    ParameterSet newpset;
    newpset.addParameter<std::string>("@service_type", service);
    addService(newpset);
  }

  void ProcessDesc::addDefaultService(std::string const& service) {
    typedef std::vector<edm::ParameterSet>::iterator Iter;
    for(Iter it = services_->begin(), itEnd = services_->end(); it != itEnd; ++it) {
      std::string name = it->getParameter<std::string>("@service_type");
      if (name == service) {
        // Use the configured service.  Don't add a default.
        return;
      }
    }
    addService(service);
  }

  void ProcessDesc::addForcedService(std::string const& service) {
    typedef std::vector<edm::ParameterSet>::iterator Iter;
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
    // Add the forced services to services_.
    for(std::vector<std::string>::const_iterator i = forcedServices.begin(), iEnd = forcedServices.end();
         i != iEnd; ++i) {
      addForcedService(*i);
    }
    // Add the default services to services_.
    for(std::vector<std::string>::const_iterator i = defaultServices.begin(), iEnd = defaultServices.end();
         i != iEnd; ++i) {
      addDefaultService(*i);
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
