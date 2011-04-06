
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <sstream>
#include <exception>

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
                                cms::Exception & iException) const {
  ParameterSet const& conf = *p.pset_;
  std::string moduleName = conf.getParameter<std::string>("@module_type");
  std::string moduleLabel = conf.getParameter<std::string>("@module_label");

  std::ostringstream ost;
  ost << "Validating configuration of module " << moduleName
      << "/'" << moduleLabel << "'";
  iException.addContext(ost.str());  
  throw;
}

void 
Maker::throwConfigurationException(ModuleDescription const& md, 
                                   sigc::signal<void, ModuleDescription const&>& post, 
                                   cms::Exception & iException) const {
  std::ostringstream ost;
  ost << "Constructing module " << md.moduleName() << "/'" << md.moduleLabel() << "'";
  iException.addContext(ost.str());
  post(md);
  throw;
}

void 
Maker::validateEDMType(std::string const& edmType, WorkerParams const& p) const {
  std::string expected = p.pset_->getParameter<std::string>("@module_edm_type");
  if (edmType != expected) {
    throw Exception(errors::Configuration)
      << "The base type in the python configuration is " << expected << ", but the base type\n"
      << "for the module's C++ class is " << edmType << ". "
      << "Please fix the configuration.\n"
      << "It must use the same base type as the C++ class.\n";
  }
}
  
std::auto_ptr<Worker> 
Maker::makeWorker(WorkerParams const& p,
                  sigc::signal<void, ModuleDescription const&>& pre,
                  sigc::signal<void, ModuleDescription const&>& post) const {
  ConfigurationDescriptions descriptions(baseType());
  fillDescriptions(descriptions);
  try {
    try {
      descriptions.validate(*p.pset_, p.pset_->getParameter<std::string>("@module_label"));    
      validateEDMType(baseType(), p);
    }
    catch (cms::Exception& e) { throw; }
    catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
    catch (std::exception& e) { convertException::stdToEDM(e); }
    catch(std::string& s) { convertException::stringToEDM(s); }
    catch(char const* c) { convertException::charPtrToEDM(c); }
    catch (...) { convertException::unknownToEDM(); }
  }
  catch (cms::Exception & iException) {
    throwValidationException(p, iException);
  }
  p.pset_->registerIt();

  ModuleDescription md = createModuleDescription(p);
  
  std::auto_ptr<Worker> worker;
  try {
    try {
      pre(md);    
      worker = makeWorker(p,md);
      post(md);
    }
    catch (cms::Exception& e) { throw; }
    catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
    catch (std::exception& e) { convertException::stdToEDM(e); }
    catch(std::string& s) { convertException::stringToEDM(s); }
    catch(char const* c) { convertException::charPtrToEDM(c); }
    catch (...) { convertException::unknownToEDM(); }
  }
  catch(cms::Exception & iException){
    throwConfigurationException(md, post, iException);
  }
  return worker;
}
  
void 
Maker::swapModule(Worker* w, ParameterSet const& p) {
   implSwapModule(w,p);
}

} // end of edm::
