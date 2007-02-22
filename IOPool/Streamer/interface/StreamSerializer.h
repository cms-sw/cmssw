#ifndef Stream_Serializer_h
#define Stream_Serializer_h

/**
 * StreamSerializer.h
 *
 * Utility class for translating framework objects (e.g. ProductRegistry and
 * EventPrincipal) into streamer message objects.
 */

#include "FWCore/Framework/interface/OutputModule.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"

class EventMsgBuilder;
class InitMsgBuilder;
namespace edm
{
  
  class EventPrincipal;
  class StreamSerializer
  {

  public:

    StreamSerializer(OutputModule::Selections const* selections);

    int serializeRegistry(InitMsgBuilder& initMessage);   
    int serializeEvent(EventPrincipal const& eventPrincipal,
                       EventMsgBuilder& eventMessage,
                       bool use_compression, int compression_level);

    static int serializeDQMEvent(DQMEvent::MonitorElementTable& meTable,
                                 DQMEventMsgBuilder& dqmMsgBuilder);

  private:

    OutputModule::Selections const* selections_;
  };

}

#endif
