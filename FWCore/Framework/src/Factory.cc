
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>
#include <memory>
#include <iostream>

using namespace std;

namespace edm {

  static void cleanup(const Factory::MakerMap::value_type& v)
  {
    delete v.second;
  }

  Factory Factory::singleInstance_;
  
  Factory::~Factory()
  {
    for_each(makers_.begin(),makers_.end(),cleanup);
  }

  Factory::Factory(): seal::PluginFactory<Maker* ()>("CMS EDM Framework Module")

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
    string modtype = p.pset_->getParameter<string>("@module_type");
    FDEBUG(1) << "Factory: module_type = " << modtype << endl;
    MakerMap::iterator it = makers_.find(modtype);

    if(it == makers_.end())
      {
	auto_ptr<Maker> wm(this->create(modtype));

	if(wm.get()==0)
	  throw edm::Exception(errors::Configuration,"UnknownModule")
	    << "Module " << modtype
	    << " with version " << p.releaseVersion_
	    << " was not registered.\n"
	    << "Perhaps your module type is misspelled or is not a "
	    << "framework plugin.\n"
	    << "Try running SealPluginDump to obtain a list of "
	    << "available Plugins.";
	  
	FDEBUG(1) << "Factory:  created worker of type " << modtype << endl;

	pair<MakerMap::iterator,bool> ret =
	  makers_.insert(make_pair<string,Maker*>(modtype,wm.get()));

	//	if(ret.second==false)
	//	  throw runtime_error("Worker Factory map insert failed");

	it = ret.first;
	wm.release();
      }

    std::auto_ptr<Worker> w(it->second->makeWorker(p,pre,post));
    return w;
  }

}
