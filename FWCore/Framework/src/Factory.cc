
#include "FWCore/CoreFramework/src/Factory.h"
#include "FWCore/CoreFramework/src/WorkerMaker.h"
#include "FWCore/CoreFramework/src/DebugMacros.h"
#include "FWCore/CoreFramework/interface/UnknownModuleException.h"

#include <utility>
#include <memory>
#include <iostream>

using namespace std;

namespace edm {

  static void cleanup(const Factory::MakerMap::value_type& v)
  {
    delete v.second;
  }

  Factory* Factory::singleInstance_=0;
  
  Factory::~Factory()
  {
    for_each(makers_.begin(),makers_.end(),cleanup);
  }

  Factory::Factory(): seal::PluginFactory<Maker* ()>("ModuleFactory")

  {
  }

  Factory* Factory::get()
  {
    //static Factory f;
    //return &f;
    if(!singleInstance_) singleInstance_=new Factory;
    return singleInstance_;
  }

  std::auto_ptr<Worker> Factory::makeWorker(ParameterSet const& conf,
					    std::string const& pn,
					    unsigned long vn,
					    unsigned long pass
					    ) const

  {
    string modtype = conf.getString("module_type");
    FDEBUG(1) << "Factory: module_type = " << modtype << endl;
    MakerMap::iterator it = makers_.find(modtype);

    if(it == makers_.end())
      {
	auto_ptr<Maker> wm(this->create(modtype));

	if(wm.get()==0) throw UnknownModuleException(modtype);
	  
	FDEBUG(1) << "Factory:  created worker of type " << modtype << endl;

	pair<MakerMap::iterator,bool> ret =
	  makers_.insert(make_pair<string,Maker*>(modtype,wm.get()));

	//	if(ret.second==false)
	//	  throw runtime_error("Worker Factory map insert failed");

	it = ret.first;
	wm.release();
      }

    std::auto_ptr<Worker> w(it->second->makeWorker(conf,pn,vn,pass));
    return w;
  }

}
