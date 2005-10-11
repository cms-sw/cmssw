#ifndef Framework_SecondaryInputSourceFactory_h
#define Framework_SecondaryInputSourceFactory_h

#include "PluginManager/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/SecondaryInputSource.h"

#include <string>
#include <memory>

namespace edm {

  typedef SecondaryInputSource* (ISFunc)(ParameterSet const&);

  class SecondaryInputSourceFactory :
    public seal::PluginFactory<ISFunc>
  {
  public:
    ~SecondaryInputSourceFactory();

    static SecondaryInputSourceFactory* get();

    std::auto_ptr<SecondaryInputSource>
      makeSecondaryInputSource(ParameterSet const&) const;
    

  private:
    SecondaryInputSourceFactory();
    static SecondaryInputSourceFactory singleInstance_;
  };

}
#endif
