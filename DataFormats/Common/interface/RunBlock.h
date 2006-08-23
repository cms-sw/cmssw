#ifndef Common_RunBlock_h
#define Common_RunBlock_h

#include "DataFormats/Common/interface/EventID.h"

// Just a placeholder for now
namespace edm {
  struct RunBlock {
    explicit RunBlock() : id_(0) {}
    RunNumber_t id_;
  };
}
#endif
