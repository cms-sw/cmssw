
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::MakerPluginFactory,"CMS EDM Framework Module");
namespace edm {

  static void cleanup(const Factory::MakerMap::value_type& v)
  {
    delete v.second;
  }

  Factory Factory::singleInstance_;
  
  Factory::~Factory()
  {
    for_all(makers_, cleanup);
  }

  Factory::Factory(): makers_()

  {
  }

  Factory* Factory::get()
  {
    return &singleInstance_;
  }

  std::auto_ptr<Worker> Factory::makeWorker(const WorkerParams& p,
                                            sigc::signal<void, const ModuleDescription&>& pre,
                                            sigc::signal<void, const ModuleDescription&>& post) const
  {
    std::string modtype = p.pset_->getParameter<std::string>("@module_type");
    FDEBUG(1) << "Factory: module_type = " << modtype << std::endl;
    MakerMap::iterator it = makers_.find(modtype);

    if(it == makers_.end())
      {
        std::auto_ptr<Maker> wm(MakerPluginFactory::get()->create(modtype));

	if(wm.get()==0)
	  throw edm::Exception(errors::Configuration,"UnknownModule")
	    << "Module " << modtype
	    << " with version " << p.processConfiguration_->releaseVersion()
	    << " was not registered.\n"
	    << "Perhaps your module type is misspelled or is not a "
	    << "framework plugin.\n"
	    << "Try running EdmPluginDump to obtain a list of "
	    << "available Plugins.";
	  
	FDEBUG(1) << "Factory:  created worker of type " << modtype << std::endl;

	std::pair<MakerMap::iterator,bool> ret =
	  makers_.insert(std::make_pair<std::string,Maker*>(modtype,wm.get()));

	//	if(ret.second==false)
	//	  throw runtime_error("Worker Factory map insert failed");

	it = ret.first;
	wm.release();
      }

    std::auto_ptr<Worker> w(it->second->makeWorker(p,pre,post));
    return w;
  }

}
