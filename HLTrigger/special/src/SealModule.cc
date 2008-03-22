#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/special/interface/HLTPixlMBFilt.h"
#include "HLTrigger/special/interface/HLTPixlMBForAlignmentFilter.h"
#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"
#include "HLTrigger/special/interface/HLTEcalIsolationFilter.h"
#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "HLTrigger/special/interface/HLTHcalPhiSymFilter.h"
#include "HLTrigger/special/interface/HLTPi0RecHitsFilter.h"
#include "HLTrigger/special/interface/HLTCSCOverlapFilter.h"
#include "HLTrigger/special/interface/HLTCSCRing2or3Filter.h"

DEFINE_FWK_MODULE(HLTPixlMBFilt);
DEFINE_FWK_MODULE(HLTPixlMBForAlignmentFilter);
DEFINE_FWK_MODULE(HLTPixelIsolTrackFilter);
DEFINE_FWK_MODULE(HLTEcalIsolationFilter);
DEFINE_FWK_MODULE(HLTEcalPhiSymFilter);
DEFINE_FWK_MODULE(HLTHcalPhiSymFilter);
DEFINE_FWK_MODULE(HLTPi0RecHitsFilter);
DEFINE_FWK_MODULE(HLTCSCOverlapFilter);
DEFINE_FWK_MODULE(HLTCSCRing2or3Filter);
