#ifndef Common_LuminosityBlock_h
#define Common_LuminosityBlock_h

#include "DataFormats/Common/interface/LuminosityBlockID.h"

// Just a placeholder for now
namespace edm {
  struct LuminosityBlock {
    explicit LuminosityBlock() : id_(0) {}
    LuminosityBlockID id_;
  };
}
#endif
