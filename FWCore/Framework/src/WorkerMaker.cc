
#include "FWCore/Framework/src/WorkerMaker.h"

namespace edm {

Maker::~Maker()
{
}

ModuleDescription 
Maker::createModuleDescription(WorkerParams const &p) const
{
  ParameterSet const& conf = *p.pset_;
  ModuleDescription md(conf.id(),
		       conf.getParameter<std::string>("@module_type"),
		       conf.getParameter<std::string>("@module_label"),
  		       p.processConfiguration_);
  return md;
}

void 
Maker::throwValidationException(WorkerParams const& p,
                                cms::Exception const& iException) const 
{
  ParameterSet const& conf = *p.pset_;
  std::string moduleName = conf.getParameter<std::string>("@module_type");
  std::string moduleLabel = conf.getParameter<std::string>("@module_label");

  edm::Exception toThrow(edm::errors::Configuration,
                         "Error occurred while validating and registering configuration\n");
  toThrow << "for module of type \'" << moduleName << "\' with label \'" << moduleLabel << "\'\n";
  toThrow.append(iException);
  throw toThrow;
}

void 
Maker::throwConfigurationException(ModuleDescription const &md, 
                                   sigc::signal<void, ModuleDescription const&>& post, 
                                   cms::Exception const& iException) const 
{
  edm::Exception toThrow(edm::errors::Configuration,"Error occurred while creating ");
  toThrow << md.moduleName() << " with label " << md.moduleLabel() << "\n";
  toThrow.append(iException);
  post(md);
  throw toThrow;
}

void 
Maker::validateEDMType(const std::string & edmType, WorkerParams const & p) const
{
  std::string expected = p.pset_->getUntrackedParameter<std::string>("@module_edm_type");
  if(edmType != expected) 
  {
    LogDebug("WorkerMaker") << "Module " << p.pset_->getParameter<std::string>("@module_label")
      << " is of type " << edmType << ", but declared in the configuration as " << expected;
  }
}


} // end of edm::
