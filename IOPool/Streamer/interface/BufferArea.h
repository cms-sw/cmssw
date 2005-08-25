#ifndef IOPOOL_BUFFER_AREA_H
#define IOPOOL_BUFFER_AREA_H

#include "IOPool/Streamer/interface/EventBuffer.h"

namespace edm {
  EventBuffer* getEventBuffer(int event_size_max, int queue_depth_max);
}

#endif
