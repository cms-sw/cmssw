
#include "FWCore/Framework/src/InputServiceFactory.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>
#include <memory>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace edm {


  InputServiceFactory::~InputServiceFactory()
  {
  }

  InputServiceFactory::InputServiceFactory(): 
    seal::PluginFactory<ISFunc>("InputServiceFactory")
  {
  }

  InputServiceFactory InputServiceFactory::singleInstance_;

  InputServiceFactory* InputServiceFactory::get()
  {
    // will not work with plugin factories
    //static InputServiceFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::auto_ptr<InputService>
  InputServiceFactory::makeInputService(ParameterSet const& conf,
					InputServiceDescription const& desc) const
    
  {
    string modtype = conf.getParameter<string>("module_type");
    FDEBUG(1) << "InputServiceFactory: module_type = " << modtype << endl;
    auto_ptr<InputService> wm(this->create(modtype,conf,desc));

    if(wm.get()==0)
      {
	throw edm::Exception(errors::Configuration,"NoSourceModule")
	  << "InputService Factory:\n"
	  << "Cannot find source type from ParameterSet: "
	  << modtype << "\n"
	  << "Perhaps your source type is mispelled or is not a SEAL Plugin?\n"
	  << "Try running SealPluginDump to obtain a list of available Plugins.";
      }

    FDEBUG(1) << "InputServiceFactory: created input service "
	      << modtype
	      << endl;

    return wm;
  }

}
