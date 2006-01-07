
#include "FWCore/Framework/src/SecondaryInputSourceFactory.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>
#include <memory>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace edm {


  SecondaryInputSourceFactory::~SecondaryInputSourceFactory()
  {
  }

  SecondaryInputSourceFactory::SecondaryInputSourceFactory(): 
    seal::PluginFactory<ISSecFunc>("CMS EDM Framework SecondaryInputSource")
  {
  }

  SecondaryInputSourceFactory SecondaryInputSourceFactory::singleInstance_;

  SecondaryInputSourceFactory* SecondaryInputSourceFactory::get()
  {
    // will not work with plugin factories
    //static SecondaryInputSourceFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::auto_ptr<SecondaryInputSource>
  SecondaryInputSourceFactory::makeSecondaryInputSource(ParameterSet const& conf) const
    
  {
    string modtype = conf.getParameter<string>("@module_type");
    FDEBUG(1) << "SecondaryInputSourceFactory: module_type = " << modtype << endl;
    auto_ptr<SecondaryInputSource> wm(this->create(modtype,conf));

    if(wm.get()==0)
      {
	throw edm::Exception(errors::Configuration,"NoSourceModule")
	  << "SecondaryInputSource Factory:\n"
	  << "Cannot find source type from ParameterSet: "
	  << modtype << "\n"
	  << "Perhaps your source type is mispelled or is not a SEAL Plugin?\n"
	  << "Try running SealPluginDump to obtain a list of available Plugins.";
      }

    FDEBUG(1) << "SecondaryInputSourceFactory: created secondary input source "
	      << modtype
	      << endl;

    return wm;
  }

}
