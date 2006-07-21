#ifndef Stream_Translator_h
#define Stream_Translator_h

/**
 * StreamTranslator.h
 *
 * Utility class for translating framework objects (e.g. ProductRegistry and
 * EventPrincipal) into streamer message objects and vice versa.
 * The "serialize" methods convert framework objects into messages, and
 * the "deserialize" methods convert messages into framework objects.
 */

#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"

namespace edm
{

  class StreamTranslator
  {

  public:

    StreamTranslator(edm::OutputModule::Selections const* selections);

    int serializeRegistry(InitMsgBuilder& initMessage);   
    int serializeEvent(EventPrincipal const& eventPrincipal,
                       EventMsgBuilder& eventMessage);

    static std::auto_ptr<SendJobHeader>
        deserializeRegistry(std::auto_ptr<InitMsgView> initView);
    static std::auto_ptr<EventPrincipal>
        deserializeEvent(std::auto_ptr<EventMsgView> eventView,
                         const ProductRegistry& productRegistry);

  private:

    edm::OutputModule::Selections const* selections_;

  };

}

#endif
