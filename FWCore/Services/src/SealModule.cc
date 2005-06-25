#include "PluginManager/ModuleDef.h"
#include "FWCore/CoreFramework/interface/InputServiceMacros.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"
#include "FWCore/FWCoreServices/src/EmptyInputService.h"
#include "FWCore/FWCoreServices/src/AsciiOutputModule.h"
#include "FWCore/FWCoreServices/src/EmptyESSource.h"
#include "FWCore/CoreFramework/interface/SourceFactory.h"

namespace edm {
  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SERVICE(EmptyInputService);
  DEFINE_ANOTHER_FWK_MODULE(AsciiOutputModule);
  DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EmptyESSource);
}
