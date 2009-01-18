/**
   \file
   Implementation of calss ProcessDesc

   \author Stefano ARGIRO
   \date 17 Jun 2005
*/


#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include <iostream>

namespace edm
{

  ProcessDesc::ProcessDesc(ParameterSet const& pset)
  : pset_(new ParameterSet(pset)), services_(new std::vector<ParameterSet>())
  {
    pset_->registerIt();
    pset::Registry::instance()->extra().setID(pset_->id());
  }

  ProcessDesc::~ProcessDesc() {
  }

  ProcessDesc::ProcessDesc(std::string const& config)
  : pset_(new ParameterSet),
    services_(new std::vector<ParameterSet>()) {
    throw edm::Exception(errors::Configuration,"Old config strings no longer accepted");
  }

  boost::shared_ptr<edm::ParameterSet>  
  ProcessDesc::getProcessPSet() const {
    return pset_;
  }

  boost::shared_ptr<std::vector<ParameterSet> > 
  ProcessDesc::getServicesPSets() const {
    return services_;
  }

  
  void ProcessDesc::addService(ParameterSet& pset) {
    services_->push_back(pset);
    pset.registerIt();
  }


  void ProcessDesc::addService(std::string const& service) {
    ParameterSet newpset;
    newpset.addParameter<std::string>("@service_type",service);
    addService(newpset);
  }

  void ProcessDesc::addDefaultService(std::string const& service) {
    typedef std::vector<edm::ParameterSet>::iterator Iter;
    for(Iter it = services_->begin(), itEnd = services_->end(); it != itEnd; ++it) {
        std::string name = it->getParameter<std::string>("@service_type");

        if (name == service) {
          // If the service is already there move it to the end so
          // it will be created before all the others already there
          // This means we use the order from the default services list
          // and the parameters from the configuration file
          while (true) {
            Iter iterNext = it + 1;
            if (iterNext == itEnd) return;
            iter_swap(it, iterNext);
            ++it;
          }
        }
    }
    addService(service);
  }


  void ProcessDesc::addServices(std::vector<std::string> const& defaultServices,
                                std::vector<std::string> const& forcedServices) {
    // Add the forced and default services to services_.
    // In services_, we want the default services first, then the forced
    // services, then the services from the configuration.  It is efficient
    // and convenient to add them in reverse order.  Then after we are done
    // adding, we reverse the std::vector again to get the desired order.
    std::reverse(services_->begin(), services_->end());
    for(std::vector<std::string>::const_reverse_iterator j = forcedServices.rbegin(),
                                            jEnd = forcedServices.rend();
         j != jEnd; ++j) {
      addService(*j);
    }
    for(std::vector<std::string>::const_reverse_iterator i = defaultServices.rbegin(),
                                            iEnd = defaultServices.rend();
         i != iEnd; ++i) {
      addDefaultService(*i);
    }
    std::reverse(services_->begin(), services_->end());
  }


} // namespace edm
