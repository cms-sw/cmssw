#ifndef FWCore_Framework_Factory_h
#define FWCore_Framework_Factory_h

#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/maker/WorkerMaker.h"
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"

#include <map>
#include <string>
#include <memory>
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edm {
  class ModuleTypeResolverBase;

  class Factory {
  public:
    typedef std::map<std::string, edm::propagate_const<Maker*>> MakerMap;

    ~Factory();

    static Factory const* get();

    //This function is not const-thread safe
    std::shared_ptr<maker::ModuleHolder> makeModule(const MakeModuleParams&,
                                                    const ModuleTypeResolverBase*,
                                                    signalslot::Signal<void(const ModuleDescription&)>& pre,
                                                    signalslot::Signal<void(const ModuleDescription&)>& post) const;

    std::shared_ptr<maker::ModuleHolder> makeReplacementModule(const edm::ParameterSet&) const;

  private:
    Factory();
    Maker* findMaker(const MakeModuleParams& p, const ModuleTypeResolverBase*) const;
    static Factory const singleInstance_;
    //It is not safe to create modules across threads
    CMS_SA_ALLOW mutable MakerMap makers_;
  };

}  // namespace edm
#endif
