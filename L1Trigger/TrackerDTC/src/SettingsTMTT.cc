#include "L1Trigger/TrackerDTC/interface/SettingsTMTT.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <cmath>

using namespace std;
using namespace edm;

namespace trackerDTC {

  SettingsTMTT::SettingsTMTT(const ParameterSet& iConfig, Settings* settings)
      :  //TrackerDTCFormat parameter sets
        paramsFormat_(iConfig.getParameter<ParameterSet>("ParamsFormat")),
        // format specific parameter
        numSectorsPhi_(paramsFormat_.getParameter<int>("NumSectorsPhi")),
        numBinsQoverPt_(paramsFormat_.getParameter<int>("NumBinsQoverPt")),
        numBinsPhiT_(paramsFormat_.getParameter<int>("NumBinsPhiT")),
        chosenRofZ_(paramsFormat_.getParameter<double>("ChosenRofZ")),
        beamWindowZ_(paramsFormat_.getParameter<double>("BeamWindowZ")),
        halfLength_(paramsFormat_.getParameter<double>("HalfLength")),
        boundariesEta_(paramsFormat_.getParameter<vector<double> >("BoundariesEta")) {
    // number of eta sectors used during track finding
    numSectorsEta_ = boundariesEta_.size() - 1;
    // cut on zT
    maxZT_ = settings->maxCot_ * chosenRofZ_;
    // width of phi sector in rad
    baseSector_ = settings->baseRegion_ / (double)numSectorsPhi_;
    // number of bits used for stub q over pt
    widthQoverPtBin_ = ceil(log2(numBinsQoverPt_));

    int& widthR = settings->widthR_;
    int& widthPhi = settings->widthPhi_;
    int& widthZ = settings->widthZ_;
    int& widthEta = settings->widthEta_;

    widthR = paramsFormat_.getParameter<int>("WidthR");
    widthPhi = paramsFormat_.getParameter<int>("WidthPhi");
    widthZ = paramsFormat_.getParameter<int>("WidthZ");
    widthEta = ceil(log2(numSectorsEta_));

    numUnusedBits_ = TTBV::S - 1 - widthR - widthPhi - widthZ - 2 * widthQoverPtBin_ - 2 * widthEta - numSectorsPhi_ -
                     settings->widthLayer_;

    double& baseQoverPt = settings->baseQoverPt_;
    double& baseR = settings->baseR_;
    double& baseZ = settings->baseZ_;
    double& basePhi = settings->basePhi_;

    baseQoverPtBin_ = settings->rangeQoverPt_ / numBinsQoverPt_;

    const int baseShiftQoverPt = widthQoverPtBin_ - settings->widthQoverPt_;

    baseQoverPt = baseQoverPtBin_ * pow(2., baseShiftQoverPt);

    const double basePhiT = baseSector_ / numBinsPhiT_;

    const double baseRgen = basePhiT / baseQoverPtBin_;
    const double rangeR = settings->outerRadius_ - settings->innerRadius_;
    const int baseShiftR = ceil(log2(rangeR / baseRgen / pow(2., widthR)));

    baseR = baseRgen * pow(2., baseShiftR);

    const double rangeZ = 2. * halfLength_;
    const int baseShiftZ = ceil(log2(rangeZ / baseR / pow(2., widthZ)));

    baseZ = baseR * pow(2., baseShiftZ);

    const double rangePhi = settings->baseRegion_ + settings->rangeQoverPt_ * baseR * pow(2., widthR) / 4.;
    const int baseShiftPhi = ceil(log2(rangePhi / basePhiT / pow(2., widthPhi)));

    basePhi = basePhiT * pow(2., baseShiftPhi);
  }

}  // namespace trackerDTC