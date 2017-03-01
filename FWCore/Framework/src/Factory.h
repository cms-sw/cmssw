#ifndef FWCore_Framework_Factory_h
#define FWCore_Framework_Factory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/MakeModuleParams.h"

#include <map>
#include <string>
#include <memory>
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {
  typedef edmplugin::PluginFactory<Maker* ()> MakerPluginFactory;
  
  class Factory  
  {
  public:
    typedef std::map<std::string, edm::propagate_const<Maker*>> MakerMap;

    ~Factory();

    static Factory const* get();

    std::shared_ptr<maker::ModuleHolder> makeModule(const MakeModuleParams&,
                                                    signalslot::Signal<void(const ModuleDescription&)>& pre,
                                                    signalslot::Signal<void(const ModuleDescription&)>& post) const;

    std::shared_ptr<maker::ModuleHolder> makeReplacementModule(const edm::ParameterSet&) const;


  private:
    Factory();
    Maker* findMaker(const MakeModuleParams& p) const;
    static Factory const singleInstance_;
    mutable MakerMap makers_;
  };

}
#endif
