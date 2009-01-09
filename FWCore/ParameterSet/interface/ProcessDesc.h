#ifndef ParameterSet_ProcessDesc_h
#define ParameterSet_ProcessDesc_h

#include "boost/shared_ptr.hpp"
#include <vector>

namespace edm
{
  
  class ParameterSet;

  class ProcessDesc
  {

  public:
    explicit ProcessDesc(const ParameterSet & pset);

    /// construct from the configuration language string
    explicit ProcessDesc(const std::string& config);

    ~ProcessDesc();

    /// get the ParameterSet that describes the process
    boost::shared_ptr<ParameterSet> getProcessPSet() const;

    /// get the dependencies for this module
    /** the return string is a list of comma-separated
      * names of the modules on which modulename depends*/
    std::string  getDependencies(const std::string& modulename);

    /// get the descriptions of the services
    boost::shared_ptr<std::vector<ParameterSet> > getServicesPSets() const;

    void addService(const ParameterSet & pset);
    /// add a service as an empty pset
    void addService(const std::string & service);
    /// add a service if it's not already there
    void addDefaultService(const std::string & service);
    /// add some defaults services, and some forced
    void addServices(std::vector<std::string> const& defaultServices,
                     std::vector<std::string> const& forcedServices);

    void setRegistry() const;

  //TODO make this private
  private:

    typedef std::vector<std::string> Strs;
    //Path and sequence information
    boost::shared_ptr<ParameterSet> pset_;
    boost::shared_ptr<std::vector< ParameterSet> > services_;
  };
}

#endif
