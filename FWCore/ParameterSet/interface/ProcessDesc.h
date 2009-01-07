#ifndef ParameterSet_ProcessDesc_h
#define ParameterSet_ProcessDesc_h

#include "boost/shared_ptr.hpp"
#include <vector>

namespace edm {
  
  class ParameterSet;

  class ProcessDesc {

  public:
    explicit ProcessDesc(ParameterSet const& pset);

    /// construct from the configuration language string
    explicit ProcessDesc(std::string const& config);

    ~ProcessDesc();

    /// get the ParameterSet that describes the process
    boost::shared_ptr<ParameterSet> getProcessPSet() const;

    /// get the dependencies for this module
    /** the return string is a list of comma-separated
      * names of the modules on which modulename depends*/
    std::string  getDependencies(std::string const& modulename);

    /// get the descriptions of the services
    boost::shared_ptr<std::vector<ParameterSet> > getServicesPSets() const;

    void addService(ParameterSet& pset);
    /// add a service as an empty pset
    void addService(std::string const& service);
    /// add a service if it's not already there
    void addDefaultService(std::string const& service);
    /// add some defaults services, and some forced
    void addServices(std::vector<std::string> const& defaultServices,
                     std::vector<std::string> const& forcedServices);

  private:

    //Path and sequence information
    boost::shared_ptr<ParameterSet> pset_;
    boost::shared_ptr<ParameterSet> trackedPartOfPset_;
    boost::shared_ptr<std::vector<ParameterSet> > services_;
  };
}

#endif
