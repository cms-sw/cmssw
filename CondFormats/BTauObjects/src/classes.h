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

    BTagEntry bte1;
    BTagEntry::Parameters bte_p1;
    std::vector<BTagEntry> v_bte1;
    std::map<std::string, std::vector<BTagEntry> > mv_bte1;
    BTagCalibration btc1;
  };
}
