#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/SiPixel/test/SiPixelCondObjBuilder.h"
#include "CondTools/SiPixel/test/SiPixelCondObjReader.h"
#include "CondTools/SiPixel/test/SiPixelCondObjForHLTBuilder.h"
#include "CondTools/SiPixel/test/SiPixelCondObjForHLTReader.h"
#include "CondTools/SiPixel/test/SiPixelCondObjOfflineBuilder.h"
#include "CondTools/SiPixel/test/SiPixelCondObjOfflineReader.h"


using cms::SiPixelCondObjBuilder;
using cms::SiPixelCondObjReader;
using cms::SiPixelCondObjForHLTBuilder;
using cms::SiPixelCondObjForHLTReader;
using cms::SiPixelCondObjOfflineBuilder;
using cms::SiPixelCondObjOfflineReader;


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelCondObjBuilder);
DEFINE_ANOTHER_FWK_MODULE(SiPixelCondObjReader);
DEFINE_ANOTHER_FWK_MODULE(SiPixelCondObjForHLTBuilder);
DEFINE_ANOTHER_FWK_MODULE(SiPixelCondObjForHLTReader);
DEFINE_ANOTHER_FWK_MODULE(SiPixelCondObjOfflineBuilder);
DEFINE_ANOTHER_FWK_MODULE(SiPixelCondObjOfflineReader);
