#ifndef Stream_Deserializer_h
#define Stream_Deserializer_h

/**
 * StreamDeserializer.h
 *
 * Utility class for translating streamer message objects into
 * framework objects (e.g. ProductRegistry and EventPrincipal)
 */

#include "DataFormats/Common/interface/ProcessConfiguration.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "TBuffer.h"
#include "TClass.h"
#include <vector>

class InitMsgView;
class EventMsgView;
namespace edm {

  class SendJobHeader;
  class EventPrincipal;
  class ProductRegistry;
  class StreamDeserializer {

  public:

    StreamDeserializer();

    static std::auto_ptr<SendJobHeader>
        deserializeRegistry(InitMsgView const& initView);
    std::auto_ptr<EventPrincipal>
        deserializeEvent(EventMsgView const& eventView,
                         ProductRegistry const& productRegistry);
    void
    setProcessConfiguration(ProcessConfiguration pc) {
      processConfiguration_ = pc;
    }

    static std::auto_ptr<DQMEvent::TObjectTable>
        deserializeDQMEvent(DQMEventMsgView const& dqmEventView);

  private:
    ProcessConfiguration processConfiguration_;
    TClass* tc_;
    std::vector<unsigned char> dest_;
    TBuffer xbuf_;
  };

}

#endif
