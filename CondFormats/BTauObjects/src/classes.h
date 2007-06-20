#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/BTauObjects/interface/CombinedTauTagCalibration.h"

namespace {

CalibratedHistogram a;
std::vector<float> b1;
std::vector<float>::iterator b1i;
std::vector<float>::const_iterator b1ic;

TrackProbabilityCategoryData c;

TrackProbabilityCalibration d;
TrackProbabilityCalibration::Entry e;
std::vector<TrackProbabilityCalibration::Entry> f;
std::vector<TrackProbabilityCalibration::Entry>::iterator f1;
std::vector<TrackProbabilityCalibration::Entry>::const_iterator f2;

CombinedTauTagCategoryData g;
CombinedTauTagCalibration h;
CombinedTauTagCalibration::Entry i;
std::vector<CombinedTauTagCalibration::Entry> j;
std::vector<CombinedTauTagCalibration::Entry>::iterator j1;
std::vector<CombinedTauTagCalibration::Entry>::const_iterator j2;
}
