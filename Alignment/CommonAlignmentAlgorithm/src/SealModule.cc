
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentProducer.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentTrackSelectorModule.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_LOOPER( AlignmentProducer );
DEFINE_ANOTHER_FWK_MODULE( AlignmentTrackSelectorModule );
