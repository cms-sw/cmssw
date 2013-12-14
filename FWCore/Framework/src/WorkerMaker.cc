
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
  Maker::createModuleDescription(MakeModuleParams const &p) const {
    ParameterSet const& conf = *p.pset_;
    ModuleDescription md(conf.id(),
                         conf.getParameter<std::string>("@module_type"),
                         conf.getParameter<std::string>("@module_label"),
                         p.processConfiguration_.get(),
                         ModuleDescription::getUniqueID());
    return md;
  }
  
  void
  Maker::throwValidationException(MakeModuleParams const& p,
                                  cms::Exception & iException) const {
    ParameterSet const& conf = *p.pset_;
    std::string moduleName = conf.getParameter<std::string>("@module_type");
    std::string moduleLabel = conf.getParameter<std::string>("@module_label");
    
    std::ostringstream ost;
    ost << "Validating configuration of module: class=" << moduleName
    << " label='" << moduleLabel << "'";
    iException.addContext(ost.str());
    throw;
  }
  
  void
  Maker::throwConfigurationException(ModuleDescription const& md,
                                     cms::Exception & iException) const {
    std::ostringstream ost;
    ost << "Constructing module: class=" << md.moduleName() << " label='" << md.moduleLabel() << "'";
    iException.addContext(ost.str());
    throw;
  }
  
  void
  Maker::validateEDMType(std::string const& edmType, MakeModuleParams const& p) const {
    std::string expected = p.pset_->getParameter<std::string>("@module_edm_type");
    if (edmType != expected) {
      throw Exception(errors::Configuration)
      << "The base type in the python configuration is " << expected << ", but the base type\n"
      << "for the module's C++ class is " << edmType << ". "
      << "Please fix the configuration.\n"
      << "It must use the same base type as the C++ class.\n";
    }
  }
  
  std::shared_ptr<maker::ModuleHolder>
  Maker::makeModule(MakeModuleParams const& p,
                    signalslot::Signal<void(ModuleDescription const&)>& pre,
                    signalslot::Signal<void(ModuleDescription const&)>& post) const {
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
    std::shared_ptr<maker::ModuleHolder> module;
    bool postCalled = false;
    try {
      try {
        pre(md);
        module = makeModule(*(p.pset_));
        module->setModuleDescription(md);
        module->preallocate(*(p.preallocate_));
        module->registerProductsAndCallbacks(p.reg_);
        // if exception then post will be called in the catch block
        postCalled = true;
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
      if(!postCalled) {
        try {
          post(md);
        }
        catch (...) {
          // If post throws an exception ignore it because we are already handling another exception
        }
      }
      throwConfigurationException(md, iException);
    }
    return module;
  }
  
  std::unique_ptr<Worker> 
  Maker::makeWorker(ExceptionToActionTable const* actions,
                    maker::ModuleHolder const* mod) const {
    
    return makeWorker(actions,mod->moduleDescription(),mod);
  }
  
} // end of edm::
