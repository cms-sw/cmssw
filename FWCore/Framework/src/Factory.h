#ifndef FWCore_Framework_Factory_h
#define FWCore_Framework_Factory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include <map>
#include <string>
#include <memory>
#include "FWCore/Utilities/interface/Signal.h"

namespace edm {
  typedef edmplugin::PluginFactory<Maker* ()> MakerPluginFactory;
  
  class Factory  
  {
  public:
    typedef std::map<std::string, Maker*> MakerMap;

    ~Factory();

    static Factory* get();

    std::auto_ptr<Worker> makeWorker(const WorkerParams&,
                                     signalslot::Signal<void(const ModuleDescription&)>& pre,
                                     signalslot::Signal<void(const ModuleDescription&)>& post) const;


  private:
    Factory();
    static Factory singleInstance_;
    mutable MakerMap makers_;
  };

}
#endif
