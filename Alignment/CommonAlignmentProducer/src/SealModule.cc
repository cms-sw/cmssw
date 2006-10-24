
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducer.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentTrackSelectorModule.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentESDataAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_LOOPER( AlignmentProducer );
DEFINE_ANOTHER_FWK_MODULE( AlignmentTrackSelectorModule );
DEFINE_ANOTHER_FWK_MODULE( AlignmentESDataAnalyzer );
