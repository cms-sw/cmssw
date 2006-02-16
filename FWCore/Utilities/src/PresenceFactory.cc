#include "FWCore/Utilities/interface/PresenceFactory.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <utility>
#include <memory>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace edm {

  PresenceFactory::~PresenceFactory() {
  }

  PresenceFactory::PresenceFactory(): 
    seal::PluginFactory<PresenceFunc>("CMS EDM Framework Presence") {
  }

  PresenceFactory PresenceFactory::singleInstance_;

  PresenceFactory* PresenceFactory::get() {
    return &singleInstance_;
  }

  std::auto_ptr<Presence>
  PresenceFactory::
  makePresence(std::string const & presence_type) const {
    auto_ptr<Presence> sp(this->create(presence_type));

    if(sp.get()==0) {
	throw edm::Exception(errors::Configuration, "NoPresenceModule")
	  << "Presence Factory:\n"
	  << "Cannot find presence type: "
	  << presence_type << "\n"
	  << "Perhaps the name is mispelled or is not a SEAL Plugin?\n"
	  << "Try running SealPluginDump to obtain a list of available Plugins.";
    }

    FDEBUG(1) << "PresenceFactory: created presence "
	      << presence_type
	      << endl;

    return sp;
  }
}
