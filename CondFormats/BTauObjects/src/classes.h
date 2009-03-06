#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/BTauObjects/interface/CombinedSVCategoryData.h"
#include "CondFormats/BTauObjects/interface/CombinedSVCalibration.h"
#include "CondFormats/BTauObjects/interface/CombinedTauTagCalibration.h"

#include "CondFormats/BTauObjects/interface/PhysicsPerformancePayload.h"


#include "CondFormats/BTauObjects/interface/BtagCorrectionPerformancePayloadFromTableEtaJetEt.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEt.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEtOnlyBeff.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEtPhi.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTable.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayload.h"
#include "CondFormats/BTauObjects/interface/BtagWorkingPoint.h"   

namespace {
  struct dictionary {
    std::vector<float> b1;

    TrackProbabilityCategoryData c;

    TrackProbabilityCalibration d;
    TrackProbabilityCalibration::Entry e;
    std::vector<TrackProbabilityCalibration::Entry> f;

    CombinedTauTagCategoryData g;
    CombinedTauTagCalibration h;
    CombinedTauTagCalibration::Entry i;
    std::vector<CombinedTauTagCalibration::Entry> j;

    CombinedSVCategoryData cs1;

    CombinedSVCalibration cs2;
    CombinedSVCalibration::Entry cs3;
    std::vector<CombinedSVCalibration::Entry> cs4;

    PhysicsPerformancePayload p1;          
    BtagCorrectionPerformancePayloadFromTableEtaJetEt c1;
    BtagPerformancePayloadFromTableEtaJetEtOnlyBeff c2;
    BtagPerformancePayloadFromTableEtaJetEt c3;
    BtagPerformancePayloadFromTableEtaJetEtPhi c4;
    // BtagPerformancePayloadFromTable c5;
    //BtagPerformancePayload c6;
    BtagWorkingPoint c7;           
  };
}
