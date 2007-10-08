#include "Alignment/CommonAlignment/plugins/AlignableMerger.h"
#include "Alignment/CommonAlignment/plugins/AlignablesFromGeometry.h"
#include "Alignment/CommonAlignment/plugins/MuonDetFromGeometry.h"
#include "Alignment/CommonAlignment/plugins/TrackerFromGeometry.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_ANOTHER_FWK_MODULE(AlignableMerger);
DEFINE_ANOTHER_FWK_MODULE(AlignablesFromGeometry);
DEFINE_ANOTHER_FWK_MODULE(MuonDetFromGeometry);
DEFINE_ANOTHER_FWK_MODULE(TrackerFromGeometry);
