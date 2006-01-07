
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>
#include <memory>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace edm {


  VectorInputSourceFactory::~VectorInputSourceFactory()
  {
  }

  VectorInputSourceFactory::VectorInputSourceFactory(): 
    seal::PluginFactory<ISVecFunc>("CMS EDM Framework VectorInputSource")
  {
  }

  VectorInputSourceFactory VectorInputSourceFactory::singleInstance_;

  VectorInputSourceFactory* VectorInputSourceFactory::get()
  {
    // will not work with plugin factories
    //static InputSourceFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::auto_ptr<VectorInputSource>
  VectorInputSourceFactory::makeVectorInputSource(ParameterSet const& conf,
					InputSourceDescription const& desc) const
    
  {
    string modtype = conf.getParameter<string>("@module_type");
    FDEBUG(1) << "VectorInputSourceFactory: module_type = " << modtype << endl;
    auto_ptr<VectorInputSource> wm(this->create(modtype,conf,desc));

    if(wm.get()==0)
      {
	throw edm::Exception(errors::Configuration,"NoSourceModule")
	  << "VectorInputSource Factory:\n"
	  << "Cannot find source type from ParameterSet: "
	  << modtype << "\n"
	  << "Perhaps your source type is mispelled or is not a SEAL Plugin?\n"
	  << "Try running SealPluginDump to obtain a list of available Plugins.";
      }

    FDEBUG(1) << "VectorInputSourceFactory: created input source "
	      << modtype
	      << endl;

    return wm;
  }

}
