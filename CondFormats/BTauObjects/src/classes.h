#include "CondFormats/BTauObjects/src/headers.h"



namespace CondFormats_BTauObjects {
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
