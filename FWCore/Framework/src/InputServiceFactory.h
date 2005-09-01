#ifndef Framework_ISFactory_h
#define Framework_ISFactory_h

#include "PluginManager/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/Framework/interface/InputService.h"

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
    static InputServiceFactory singleInstance_;
  };

}
#endif
