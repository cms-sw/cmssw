#ifndef L1Trigger_TrackFindingTMTT_Sector_h
#define L1Trigger_TrackFindingTMTT_Sector_h

#include "L1Trigger/TrackFindingTMTT/interface/TP.h"

#include <vector>
#include <unordered_map>

namespace tmtt {

  class Settings;
  class Stub;

  class Sector {
  public:
    // Initialization.
    Sector(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaSec);

    // Check if stub within the eta and/or phi boundaries of this sector.
    bool inside(const Stub* stub) const { return (this->insideEta(stub) && this->insidePhi(stub)); }
    bool insideEta(const Stub* stub) const;
    bool insidePhi(const Stub* stub) const;

    // Check if stub is within subsectors in eta that sector may be divided into.
    std::vector<bool> insideEtaSubSecs(const Stub* stub) const;

    unsigned int iPhiSec() const { return iPhiSec_; }  // Sector number.
    unsigned int iEtaReg() const { return iEtaReg_; }
    float phiCentre() const { return phiCentre_; }  // Return phi of centre of this sector.
    float etaMin() const { return etaMin_; }        // Eta range covered by this sector.
    float etaMax() const { return etaMax_; }        // Eta range covered by this sector.

    float sectorHalfWidth() const { return sectorHalfWidth_; }  // Half width in phi of sector measured in radians.
    float zAtChosenR_Min() const {
      return zOuterMin_;
    }  // Range in z of particle at chosen radius from beam line covered by this sector.
    float zAtChosenR_Max() const { return zOuterMax_; }

    // For performance studies, note which stubs on given tracking particle are inside the sector.
    // Returns two booleans for each stub, indicating if they are in phi & eta sectors respectively.
    // You can AND them together to check if stub is in (eta,phi) sector.
    std::unordered_map<const Stub*, std::pair<bool, bool>> stubsInside(const TP& tp) const;

    // Count number of stubs in given tracking particle which are inside this (phi,eta) sector;
    // or inside it if only the eta cuts are applied; or inside it if only the phi cuts are applied.
    // The results are returned as the 3 last arguments of the function.
    void numStubsInside(const TP& tp,
                        unsigned int& nStubsInsideEtaPhi,
                        unsigned int& nStubsInsideEta,
                        unsigned int& nStubsInsidePhi) const;

    // Check if the helix parameters of a tracking particle (truth) are consistent with this sector.
    bool insidePhiSec(const TP& tp) const {
      return (std::abs(tp.trkPhiAtR(chosenRofPhi_) - phiCentre_) < sectorHalfWidth_);
    }
    bool insideEtaReg(const TP& tp) const {
      return (tp.trkZAtR(chosenRofZ_) > zOuterMin_ && tp.trkZAtR(chosenRofZ_) < zOuterMax_);
    }

  private:
    // Check if stub is within eta sector or subsector that is delimated by specified zTrk range.
    bool insideEtaRange(const Stub* stub, float zRangeMin, float zRangeMax) const;

    // Digitize a floating point number to 2s complement integer, dropping anything after the decimal point. (Kristian Harder)
    int64_t forceBitWidth(const float value, const UInt_t nBits) const;

    // Check if stub is within subsectors in eta that sector may be divided into. Uses digitized calculation corresponding to GP firmware. (Kristian Harder)
    std::vector<bool> subEtaFwCalc(const int rT, const int z) const;

  private:
    const Settings* settings_;

    // Sector number
    unsigned int iPhiSec_;
    unsigned int iEtaReg_;
    float beamWindowZ_;
    float etaMin_;  // Range in eta covered by this sector.
    float etaMax_;
    float chosenRofZ_;  // Use z of track at radius="chosenRofZ" to define eta sectors.
    float zOuterMin_;   // z range of sector at reference radius
    float zOuterMax_;

    // Define phi sector.
    float phiCentre_;         // phi of centre of sector.
    float sectorHalfWidth_;   // sector half-width excluding overlaps.
    float chosenRofPhi_;      // Use phi of track at radius="chosenRofPhi" to define phi sectors.
    float minPt_;             // Min Pt covered by HT array.
    float assumedPhiTrkRes_;  // Tolerance in stub phi0 (or phi65) assumed to be this fraction of phi sector width. (N.B. If > 0.5, then stubs can be shared by more than 2 phi sectors).
    bool useStubPhi_;  // Require stub phi to be consistent with track of Pt > HTArraySpec.HoughMinPt that crosses HT phi axis?
    bool useStubPhiTrk_;  // Require stub phi0 (or phi65 etc.) as estimated from stub bend, to lie within HT phi axis, allowing tolerance specified below?
    bool calcPhiTrkRes_;  // If true, tolerance in stub phi0 (or phi65 etc.) will be reduced below AssumedPhiTrkRes if stub bend resolution specified in HTFilling.BendResolution suggests it is safe to do so.

    // Possible subsectors in eta within each sector.
    unsigned int numSubSecsEta_;
    std::vector<float> zOuterMinSub_;
    std::vector<float> zOuterMaxSub_;
  };

}  // namespace tmtt

#endif
