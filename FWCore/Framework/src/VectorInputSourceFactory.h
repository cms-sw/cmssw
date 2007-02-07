#ifndef Framework_VectorInputSourceFactory_h
#define Framework_VectorInputSourceFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#include <string>
#include <memory>

namespace edm {

  typedef VectorInputSource* (ISVecFunc)(ParameterSet const&, InputSourceDescription const&);

  class VectorInputSourceFactory :
    public seal::PluginFactory<ISVecFunc>
  {
  public:
    ~VectorInputSourceFactory();

    static VectorInputSourceFactory* get();

    std::auto_ptr<VectorInputSource>
      makeVectorInputSource(ParameterSet const&,
		       InputSourceDescription const&) const;
    

  private:
    VectorInputSourceFactory();
    static VectorInputSourceFactory singleInstance_;
  };

}
#endif
