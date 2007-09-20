#ifndef IOPool_Streamer_StreamDQMDeserializer_h
#define IOPool_Streamer_StreamDQMDeserializer_h

/**
 * StreamDQMDeserializer.h
 *
 * Utility class for translating streamer message objects into
 * DQM objects (monitor elements)
 */

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "TBuffer.h"
#include <vector>

namespace edm {

  class StreamDQMDeserializer {

  public:
    StreamDQMDeserializer();

    std::auto_ptr<DQMEvent::TObjectTable>
      deserializeDQMEvent(DQMEventMsgView const& dqmEventView);

  private:
    std::vector<unsigned char> decompressBuffer_;
    TBuffer workTBuffer_;
  };

}

#endif
