#ifndef Streamer_InputSource_h
#define Streamer_InputSource_h

#include "boost/shared_ptr.hpp"

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "IOPool/Streamer/interface/StreamDeserializer.h"

namespace edm {
  class StreamerInputSource : public InputSource {
  public:  
    explicit StreamerInputSource(ParameterSet const& pset,
                 InputSourceDescription const& desc);
    virtual ~StreamerInputSource();

    std::auto_ptr<SendJobHeader> deserializeRegistry(InitMsgView const& initView) {
      return deserializer_.deserializeRegistry(initView);
    }

    std::auto_ptr<EventPrincipal> deserializeEvent(EventMsgView const& eventView, boost::shared_ptr<ProductRegistry const> preg) {
      return deserializer_.deserializeEvent(eventView, preg);
    }

    void mergeWithRegistry(SendDescs const& descs, ProductRegistry&);

    static void mergeIntoRegistry(SendDescs const& descs, ProductRegistry&);

  protected:
    void declareStreamers(SendDescs const& descs);
    void buildClassCache(SendDescs const& descs);

  private:
    StreamDeserializer deserializer_;
  }; //end-of-class-def
} // end of namespace-edm
  
#endif
