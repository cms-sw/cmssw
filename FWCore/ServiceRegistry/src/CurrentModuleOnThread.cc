#include "FWCore/ServiceRegistry/interface/CurrentModuleOnThread.h"

thread_local edm::ModuleCallingContext const* edm::CurrentModuleOnThread::currentModuleOnThread_ = nullptr;
