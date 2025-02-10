#include "PauseMaxMemoryPreloadSentry.h"

// By default do nothing, but add "hooks" that MaxMemoryPreload can
// override with LD_PRELOAD
void pauseMaxMemoryPreload() {}
void unpauseMaxMemoryPreload() {}

namespace edm {
  PauseMaxMemoryPreloadSentry::PauseMaxMemoryPreloadSentry() { pauseMaxMemoryPreload(); }
  PauseMaxMemoryPreloadSentry::~PauseMaxMemoryPreloadSentry() { unpauseMaxMemoryPreload(); }
}  // namespace edm
