#ifndef IOPool_Streamer_StreamDQMDeserializer_h
#define IOPool_Streamer_StreamDQMDeserializer_h

/**
 * StreamDQMDeserializer.h
 *
 * Utility class for translating streamer message objects into
 * DQM objects (monitor elements)
 */


#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
#include "TBufferFile.h"
typedef TBufferFile RootBuffer;
#else
#include "TBuffer.h"
typedef TBuffer RootBuffer;
#endif

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include <vector>

namespace edm {

  class StreamDQMDeserializer {

  public:
    StreamDQMDeserializer();

    std::auto_ptr<DQMEvent::TObjectTable>
      deserializeDQMEvent(DQMEventMsgView const& dqmEventView);

  private:
    std::vector<unsigned char> decompressBuffer_;
    RootBuffer workTBuffer_;
  };

}

#endif
