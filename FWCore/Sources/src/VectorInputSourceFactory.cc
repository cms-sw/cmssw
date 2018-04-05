
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::VectorInputSourcePluginFactory,"CMS EDM Framework VectorInputSource");

namespace edm {

  VectorInputSourceFactory::~VectorInputSourceFactory() {
  }

  VectorInputSourceFactory::VectorInputSourceFactory() {
  }

  VectorInputSourceFactory const VectorInputSourceFactory::singleInstance_;

  VectorInputSourceFactory const* VectorInputSourceFactory::get() {
    // will not work with plugin factories
    //static InputSourceFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::unique_ptr<VectorInputSource>
  VectorInputSourceFactory::makeVectorInputSource(ParameterSet const& conf,
					VectorInputSourceDescription const& desc) const {
    std::string modtype = conf.getParameter<std::string>("@module_type");
    FDEBUG(1) << "VectorInputSourceFactory: module_type = " << modtype << std::endl;
    std::unique_ptr<VectorInputSource> wm(VectorInputSourcePluginFactory::get()->create(modtype,conf,desc));

    if(wm.get() == nullptr) {
	throw edm::Exception(errors::Configuration,"NoSourceModule")
	  << "VectorInputSource Factory:\n"
	  << "Cannot find source type from ParameterSet: "
	  << modtype << "\n"
	  << "Perhaps your source type is misspelled or is not an EDM Plugin?\n"
	  << "Try running EdmPluginDump to obtain a list of available Plugins.";
    }

    FDEBUG(1) << "VectorInputSourceFactory: created input source "
	      << modtype
	      << std::endl;

    return wm;
  }

}
