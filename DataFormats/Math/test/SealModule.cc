#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Math/test/WriteMath.h"
#include "DataFormats/Math/test/ReadMath.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( WriteMath );
DEFINE_ANOTHER_FWK_MODULE( ReadMath );
