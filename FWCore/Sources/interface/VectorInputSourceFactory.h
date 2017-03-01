#ifndef FWCore_Sources_VectorInputSourceFactory_h
#define FWCore_Sources_VectorInputSourceFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Sources/interface/VectorInputSource.h"

#include <memory>
#include <string>

namespace edm {
  struct VectorInputSourceDescription;
  class ParameterSet;

  typedef VectorInputSource* (ISVecFunc)(ParameterSet const&, VectorInputSourceDescription const&);
  typedef edmplugin::PluginFactory<ISVecFunc> VectorInputSourcePluginFactory;

  class VectorInputSourceFactory {
  public:
    ~VectorInputSourceFactory();

    static VectorInputSourceFactory const* get();

    std::unique_ptr<VectorInputSource>
      makeVectorInputSource(ParameterSet const&,
                            VectorInputSourceDescription const&) const;

  private:
    VectorInputSourceFactory();
    static VectorInputSourceFactory const singleInstance_;
  };
}
#endif
