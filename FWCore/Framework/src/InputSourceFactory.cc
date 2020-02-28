
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::InputSourcePluginFactory, "CMS EDM Framework InputSource");
namespace edm {

  InputSourceFactory::~InputSourceFactory() {}

  InputSourceFactory::InputSourceFactory() {}

  InputSourceFactory const InputSourceFactory::singleInstance_;

  InputSourceFactory const* InputSourceFactory::get() {
    // will not work with plugin factories
    //static InputSourceFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::unique_ptr<InputSource> InputSourceFactory::makeInputSource(ParameterSet const& conf,
                                                                   InputSourceDescription const& desc) const

  {
    std::string modtype = conf.getParameter<std::string>("@module_type");
    FDEBUG(1) << "InputSourceFactory: module_type = " << modtype << std::endl;
    auto wm = InputSourcePluginFactory::get()->create(modtype, conf, desc);

    if (wm.get() == nullptr) {
      throw edm::Exception(errors::Configuration, "NoSourceModule")
          << "InputSource Factory:\n"
          << "Cannot find source type from ParameterSet: " << modtype << "\n"
          << "Perhaps your source type is misspelled or is not an EDM Plugin?\n"
          << "Try running EdmPluginDump to obtain a list of available Plugins.";
    }

    wm->registerProducts();

    FDEBUG(1) << "InputSourceFactory: created input source " << modtype << std::endl;

    return wm;
  }

}  // namespace edm
