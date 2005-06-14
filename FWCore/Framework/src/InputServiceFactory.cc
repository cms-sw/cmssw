
#include "FWCore/CoreFramework/src/InputServiceFactory.h"
#include "FWCore/CoreFramework/src/WorkerMaker.h"
#include "FWCore/CoreFramework/src/DebugMacros.h"

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

  InputServiceFactory* InputServiceFactory::get()
  {
    static InputServiceFactory f;
    return &f;
  }

  std::auto_ptr<InputService>
  InputServiceFactory::makeInputService(ParameterSet const& conf,
					InputServiceDescription const& desc
					) const
    
  {
    string modtype = getParameter<string>(conf, "module_type");
    FDEBUG(1) << "InputServiceFactory: module_type = " << modtype << endl;
    auto_ptr<InputService> wm(this->create(modtype,conf,desc));

    if(wm.get()==0)
      {
	string tmp("InputService Factory:\n Cannot find source type from ParameterSet: ");
	tmp+=modtype;
	tmp+="\n Perhaps your source type is mispelled or is not a SEAL Plugin?";
	tmp+="\n Try running SealPluginDump to obtain a list of available Plugins.";
	throw runtime_error(tmp);
      }

    FDEBUG(1) << "InputServiceFactory: created input service "
	      << modtype
	      << endl;

    return wm;
  }

}
