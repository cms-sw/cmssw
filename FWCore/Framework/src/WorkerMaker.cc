
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

Maker::~Maker() {
}

ModuleDescription 
Maker::createModuleDescription(WorkerParams const &p) const {
  ParameterSet const& conf = *p.pset_;
  ModuleDescription md(conf.id(),
		       conf.getParameter<std::string>("@module_type"),
		       conf.getParameter<std::string>("@module_label"),
  		       p.processConfiguration_);
  return md;
}

void 
Maker::throwValidationException(WorkerParams const& p,
                                cms::Exception const& iException) const {
  ParameterSet const& conf = *p.pset_;
  std::string moduleName = conf.getParameter<std::string>("@module_type");
  std::string moduleLabel = conf.getParameter<std::string>("@module_label");

  Exception toThrow(errors::Configuration,
                         "Error occurred while validating and registering configuration\n");
  toThrow << "for module of type \'" << moduleName << "\' with label \'" << moduleLabel << "\'\n";
  toThrow.append(iException);
  throw toThrow;
}

void 
Maker::throwConfigurationException(ModuleDescription const& md, 
                                   sigc::signal<void, ModuleDescription const&>& post, 
                                   cms::Exception const& iException) const {
  Exception toThrow(errors::Configuration,"Error occurred while creating ");
  toThrow << "for module of type \'"<<md.moduleName() << "\' with label \'" << md.moduleLabel() << "'\n";
  toThrow.append(iException);
  post(md);
  throw toThrow;
}

void 
Maker::validateEDMType(std::string const& edmType, WorkerParams const& p) const {
  std::string expected = p.pset_->getParameter<std::string>("@module_edm_type");
  if(edmType != expected) {
    Exception toThrow(errors::Configuration,"Error occurred while creating module.\n");
    toThrow <<"Module of type \'"<<  p.pset_->getParameter<std::string>("@module_type") <<  "' with label '" << p.pset_->getParameter<std::string>("@module_label")
      << "' is of type " << edmType << ", but declared in the configuration as " << expected << ".\n"
      << "Please replace " << expected << " with " << edmType << " in the appropriate configuration file(s).\n";
    throw toThrow;
  }
}
  
std::auto_ptr<Worker> 
Maker::makeWorker(WorkerParams const& p,
                  sigc::signal<void, ModuleDescription const&>& pre,
                  sigc::signal<void, ModuleDescription const&>& post) const {
  try {
    ConfigurationDescriptions descriptions(baseType());
    fillDescriptions(descriptions);
    descriptions.validate(*p.pset_, p.pset_->getParameter<std::string>("@module_label"));    
    p.pset_->registerIt();
  }
  catch (cms::Exception& iException) {
    throwValidationException(p, iException);
  }
  
  ModuleDescription md = createModuleDescription(p);
  
  std::auto_ptr<Worker> worker;
  validateEDMType(baseType(), p);
  try {
    pre(md);    
    worker = makeWorker(p,md);

    post(md);
  } catch( cms::Exception& iException){
    throwConfigurationException(md, post, iException);
  }
  return worker;
}
  
void 
Maker::swapModule(Worker* w, ParameterSet const& p) {
   implSwapModule(w,p);
}

} // end of edm::
