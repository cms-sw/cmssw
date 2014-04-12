#ifndef FWCore_Framework_InputSourceFactory_h
#define FWCore_Framework_InputSourceFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <string>
#include <memory>

namespace edm {

  typedef InputSource* (ISFunc)(ParameterSet const&, InputSourceDescription const&);

  typedef edmplugin::PluginFactory<ISFunc> InputSourcePluginFactory;

  class InputSourceFactory {
  public:
    ~InputSourceFactory();

    static InputSourceFactory const* get();

    std::auto_ptr<InputSource>
      makeInputSource(ParameterSet const&,
		       InputSourceDescription const&) const;
    

  private:
    InputSourceFactory();
    static InputSourceFactory const singleInstance_;
  };

}
#endif
