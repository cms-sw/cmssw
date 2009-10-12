#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/BTauObjects/interface/CombinedSVCategoryData.h"
#include "CondFormats/BTauObjects/interface/CombinedSVCalibration.h"
#include "CondFormats/BTauObjects/interface/CombinedTauTagCalibration.h"


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

  };
}
