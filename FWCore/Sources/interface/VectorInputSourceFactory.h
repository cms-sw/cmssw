#ifndef FWCore_Sources_VectorInputSourceFactory_h
#define FWCore_Sources_VectorInputSourceFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Sources/interface/VectorInputSource.h"

#include <memory>
#include <string>

namespace edm {
  struct InputSourceDescription;
  class ParameterSet;

  typedef VectorInputSource* (ISVecFunc)(ParameterSet const&, InputSourceDescription const&);
  typedef edmplugin::PluginFactory<ISVecFunc> VectorInputSourcePluginFactory;

  class VectorInputSourceFactory {
  public:
    ~VectorInputSourceFactory();

    static VectorInputSourceFactory* get();

    std::unique_ptr<VectorInputSource>
      makeVectorInputSource(ParameterSet const&,
                            InputSourceDescription const&) const;

  private:
    VectorInputSourceFactory();
    static VectorInputSourceFactory singleInstance_;
  };
}
#endif
