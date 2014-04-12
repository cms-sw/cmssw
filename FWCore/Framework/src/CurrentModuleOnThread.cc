#include "FWCore/Framework/interface/CurrentModuleOnThread.h"

thread_local edm::ModuleCallingContext const* edm::CurrentModuleOnThread::currentModuleOnThread_ = nullptr;
