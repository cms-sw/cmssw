#ifndef EDM_ISFACTORY_H
#define EDM_ISFACTORY_H

#include "PluginManager/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/CoreFramework/interface/InputServiceDescription.h"
#include "FWCore/CoreFramework/interface/InputService.h"

#include <string>
#include <memory>

namespace edm {

  typedef InputService* (ISFunc)(ParameterSet const&, InputServiceDescription const&);

  class InputServiceFactory :
    public seal::PluginFactory<ISFunc>
  {
  public:
    ~InputServiceFactory();

    static InputServiceFactory* get();

    std::auto_ptr<InputService>
      makeInputService(ParameterSet const&,
		       InputServiceDescription const&) const;
    

  private:
    InputServiceFactory();
  };

}
#endif
