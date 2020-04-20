#ifndef L1Trigger_TrackerDTC_SettingsTMTT_h
#define L1Trigger_TrackerDTC_SettingsTMTT_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace trackerDTC {

  class Settings;

  // TMTT data format specific settings
  class SettingsTMTT {
    friend class Settings;

  public:
    SettingsTMTT(const edm::ParameterSet& iConfig, Settings* settings);
    ~SettingsTMTT() {}

    // format specific parameter

    int numSectorsPhi() const { return numSectorsPhi_; }
    int numBinsQoverPt() const { return numBinsQoverPt_; }
    int numBinsPhiT() const { return numBinsPhiT_; }
    double chosenRofZ() const { return chosenRofZ_; }
    double beamWindowZ() const { return beamWindowZ_; }
    double boundariesEta(const int& eta) const { return boundariesEta_[eta]; }

    // format specific parameter

    int numSectorsEta() const { return numSectorsEta_; }
    int widthQoverPtBin() const { return widthQoverPtBin_; }
    int numUnusedBits() const { return numUnusedBits_; }
    double maxZT() const { return maxZT_; }
    double baseSector() const { return baseSector_; }
    double baseQoverPt() const { return baseQoverPtBin_; }

  private:
    //TrackerDTCFormat parameter sets

    const edm::ParameterSet paramsFormat_;

    // format specific parameter

    // number of phi sectors used during track finding
    const int numSectorsPhi_;
    // number of qOverPt bins used during track finding
    const int numBinsQoverPt_;
    // number of phiT bins used during track finding
    const int numBinsPhiT_;
    // critical radius defining r-z sector shape in cm
    const double chosenRofZ_;
    // half lumi region size in cm
    const double beamWindowZ_;
    // has to be >= max stub z / 2 in cm
    const double halfLength_;
    // defining r-z sector shape
    const std::vector<double> boundariesEta_;

    // derived format specific parameter

    // number of eta sectors used during track finding
    int numSectorsEta_;
    // number of bits used for stub q over pt
    int widthQoverPtBin_;
    // number of padded 0s in output data format
    int numUnusedBits_;
    // cut on zT
    double maxZT_;
    // width of phi sector in rad
    double baseSector_;
    // precision of qOverPt bins used during track finding
    double baseQoverPtBin_;
  };

}  // namespace trackerDTC

#endif