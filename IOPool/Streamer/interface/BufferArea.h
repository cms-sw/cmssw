#ifndef Streamer_BufferArea_h
#define Streamer_BufferArea_h

// -*- C++ -*-

#include "IOPool/Streamer/interface/EventBuffer.h"

namespace edm {
  EventBuffer* getEventBuffer(int event_size_max, int queue_depth_max);
}

#endif
