#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;

namespace tmtt {

  //=== Initialise

  Sector::Sector(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg)
      : settings_(settings),
        // Sector number
        iPhiSec_(iPhiSec),
        iEtaReg_(iEtaReg),

        beamWindowZ_(settings->beamWindowZ()),  // Assumed half-length of beam-spot

        //===  Characteristics of this eta region.
        // Using lines of specified rapidity drawn from centre of CMS, determine the z coords at which
        // they cross the radius chosenRofZ_.
        etaMin_(settings->etaRegions()[iEtaReg]),
        etaMax_(settings->etaRegions()[iEtaReg + 1]),
        chosenRofZ_(settings->chosenRofZ()),
        // Get range in z of tracks covered by this sector at chosen radius from beam-line
        zOuterMin_(chosenRofZ_ / tan(2. * atan(exp(-etaMin_)))),
        zOuterMax_(chosenRofZ_ / tan(2. * atan(exp(-etaMax_)))),

        //=== Characteristics of this phi region.
        chosenRofPhi_(settings->chosenRofPhi()),
        minPt_(settings->houghMinPt()),  // Min Pt covered by  HT array.
        assumedPhiTrkRes_(settings->assumedPhiTrkRes()),
        useStubPhi_(settings->useStubPhi()),
        useStubPhiTrk_(settings->useStubPhiTrk()),
        calcPhiTrkRes_(settings->calcPhiTrkRes()),
        //=== Check if subsectors in eta are being used within each sector.
        numSubSecsEta_(settings->numSubSecsEta()) {
    // Centre of phi (tracking) nonant zero must be along x-axis to be consistent with tracker cabling map.
    // Define phi sector zero  to start at lower end of phi range in nonant 0.
    float phiCentreSec0 = -M_PI / float(settings->numPhiNonants()) + M_PI / float(settings->numPhiSectors());
    // Centre of sector in phi
    phiCentre_ = 2. * M_PI * float(iPhiSec) / float(settings->numPhiSectors()) + phiCentreSec0;
    sectorHalfWidth_ = M_PI / float(settings->numPhiSectors());  // Sector half width excluding overlaps.

    // If eta subsectors have equal width in rapidity, do this.
    float subSecWidth = (etaMax_ - etaMin_) / float(numSubSecsEta_);
    for (unsigned int i = 0; i < numSubSecsEta_; i++) {
      float subSecEtaMin = etaMin_ + i * subSecWidth;
      float subSecEtaMax = subSecEtaMin + subSecWidth;
      float subSecZmin = chosenRofZ_ / tan(2. * atan(exp(-subSecEtaMin)));
      float subSecZmax = chosenRofZ_ / tan(2. * atan(exp(-subSecEtaMax)));
      zOuterMinSub_.push_back(subSecZmin);
      zOuterMaxSub_.push_back(subSecZmax);
    }
  }

  //=== Check if stub is inside this eta region.

  bool Sector::insideEta(const Stub* stub) const {
    // Lower edge of this eta region defined by line from (r,z) = (0,-beamWindowZ) to (chosenRofZ_, zOuterMin_).
    // Upper edge of this eta region defined by line from (r,z) = (0, beamWindowZ) to (chosenRofZ_, zOuterMax_).

    bool inside = this->insideEtaRange(stub, zOuterMin_, zOuterMax_);
    return inside;
  }

  //=== Check if stub is within subsectors in eta that sector may be divided into.

  vector<bool> Sector::insideEtaSubSecs(const Stub* stub) const {
    if (settings_->enableDigitize() && numSubSecsEta_ == 2) {
      // Use (complicated) digitized firmware emulation
      return subEtaFwCalc(stub->digitalStub()->iDigi_Rt(), stub->digitalStub()->iDigi_Z());

    } else {
      // Use (simpler) floating point calculation.

      vector<bool> insideVec;

      // Loop over subsectors.
      for (unsigned int i = 0; i < numSubSecsEta_; i++) {
        bool inside = this->insideEtaRange(stub, zOuterMinSub_[i], zOuterMaxSub_[i]);
        insideVec.push_back(inside);
      }

      return insideVec;
    }
  }

  //=== Check if stub is within eta sector or subsector that is delimated by specified zTrk range.

  bool Sector::insideEtaRange(const Stub* stub, float zRangeMin, float zRangeMax) const {
    // Lower edge of this eta region defined by line from (r,z) = (0,-beamWindowZ) to (chosenRofZ_, zRangeMin).
    // Upper edge of this eta region defined by line from (r,z) = (0, beamWindowZ) to (chosenRofZ_, zRangeMax).

    float zMin, zMax;
    bool inside;

    // Calculate z coordinate of lower edge of this eta region, evaluated at radius of stub.
    zMin = (zRangeMin * stub->r() - beamWindowZ_ * std::abs(stub->r() - chosenRofZ_)) / chosenRofZ_;
    // Calculate z coordinate of upper edge of this eta region, evaluated at radius of stub.
    zMax = (zRangeMax * stub->r() + beamWindowZ_ * std::abs(stub->r() - chosenRofZ_)) / chosenRofZ_;

    inside = (stub->z() > zMin && stub->z() < zMax);
    return inside;
  }

  //=== Check if stub is inside this phi region.

  bool Sector::insidePhi(const Stub* stub) const {
    // N.B. The logic here for preventing a stub being assigned to > 2 sectors seems overly agressive.
    // But attempts at improving it have failed ...

    bool okPhi = true;
    bool okPhiTrk = true;

    if (useStubPhi_) {
      float delPhi =
          reco::deltaPhi(stub->phi(), phiCentre_);  // Phi difference between stub & sector in range -PI to +PI.
      float tolerancePhi = stub->phiDiff(
          chosenRofPhi_, minPt_);  // How much stub phi might differ from track phi because of track curvature.
      float outsidePhi = std::abs(delPhi) - sectorHalfWidth_ -
                         tolerancePhi;  // If > 0, then stub is not compatible with being inside this sector.
      if (outsidePhi > 0)
        okPhi = false;
    }

    if (useStubPhiTrk_) {
      // Estimate either phi0 of track from stub info, or phi of the track at radius chosenRofPhi_.
      float phiTrk = stub->trkPhiAtR(chosenRofPhi_);
      // Phi difference between stub & sector in range -PI to +PI.
      float delPhiTrk = reco::deltaPhi(phiTrk, phiCentre_);
      // Set tolerance equal to nominal resolution assumed in phiTrk
      float tolerancePhiTrk = assumedPhiTrkRes_ * (2 * sectorHalfWidth_);
      if (calcPhiTrkRes_) {
        // Calculate uncertainty in phiTrk due to poor resolution in stub bend
        float phiTrkRes = stub->trkPhiAtRcut(chosenRofPhi_);
        // Reduce tolerance if this is smaller than the nominal assumed resolution.
        tolerancePhiTrk = min(tolerancePhiTrk, phiTrkRes);
      }
      // If following > 0, then stub is not compatible with being inside this sector.
      float outsidePhiTrk = std::abs(delPhiTrk) - sectorHalfWidth_ - tolerancePhiTrk;

      if (outsidePhiTrk > 0)
        okPhiTrk = false;
    }

    return (okPhi && okPhiTrk);
  }

  //=== For performance studies, note which stubs on given tracking particle are inside the sector.
  //=== Returns two booleans for each stub, indicating if they are in phi & eta sectors respectively.
  //=== AND them together to get (eta,phi) sector decision.

  unordered_map<const Stub*, pair<bool, bool> > Sector::stubsInside(const TP& tp) const {
    unordered_map<const Stub*, pair<bool, bool> > inside;
    // Loop over stubs produced by tracking particle
    const vector<const Stub*>& assStubs = tp.assocStubs();
    for (const Stub* stub : assStubs) {
      // Check if this stub is inside sector
      inside[stub] = pair<bool, bool>(this->insidePhi(stub), this->insideEta(stub));
    }
    return inside;
  }

  //=== Count number of stubs in given tracking particle which are inside this (phi,eta) sector;
  //=== or inside it if only the eta cuts are applied; or inside it if only the phi cuts are applied.
  //=== The results are returned as the 3 last arguments of the function.

  void Sector::numStubsInside(const TP& tp,
                              unsigned int& nStubsInsideEtaPhi,
                              unsigned int& nStubsInsideEta,
                              unsigned int& nStubsInsidePhi) const {
    nStubsInsideEtaPhi = 0;
    nStubsInsideEta = 0;
    nStubsInsidePhi = 0;
    for (const auto& iter : this->stubsInside(tp)) {
      bool insidePhi = iter.second.first;
      bool insideEta = iter.second.second;
      if (insidePhi && insideEta)
        nStubsInsideEtaPhi++;
      if (insideEta)
        nStubsInsideEta++;
      if (insidePhi)
        nStubsInsidePhi++;
    }
  }

  // Digitize a floating point number to 2s complement integer, dropping anything after the decimal point. (Kristian Harder)

  int64_t Sector::forceBitWidth(const float value, const UInt_t nBits) const {
    // slightly hand-waving treatment of 2s complement
    int64_t sign = 1;
    if (value < 0)
      sign = -1;
    int64_t iValue = int64_t(std::abs(value));
    int64_t mask = (int64_t(1) << nBits) - int64_t(1);
    int64_t result = sign * (iValue & mask);
    if (std::abs(result - value) > 1)
      throw cms::Exception("LogicError")
          << "Sector::forceBitWidth is messing up by using too few bits to digitize number"
          << " nBits=" << nBits << " Input float=" << value << " Output digi = " << result;
    return result;
    // Check that result is compatible with value. Throw error if not.
  }

  //=== Check if stub is within subsectors in eta that sector may be divided into. Uses digitized calculation corresponding to GP firmware. (Kristian Harder)
  //=== Modified to configurable number of rT and z digisation bits by Ian, with advice from Luis.

  vector<bool> Sector::subEtaFwCalc(const int rT, const int z) const {
    // Note number of reference bits used to digitize rT and z, used when GP authors determined some constants below.
    unsigned int rtBitsRef = 10;
    unsigned int zBitsRef = 12;

    // This replaces Kristian's hard-wired constants with configurable ones.
    unsigned int rtBits = settings_->rtBits();
    unsigned int zBits = settings_->zBits();
    float rtRange = settings_->rtRange();
    float zRange = settings_->zRange();
    constexpr float cm_to_mm = 10.;  // firwmare is in mm and CMSSW in cm.
    float zBase = cm_to_mm / (pow(2, zBits) / zRange);
    float rTBase = cm_to_mm / (pow(2, rtBits) / rtRange);

    // Number of bits used by DSP in UltraScale-Plus FPGA (where DSP does D = A*B + C)
    constexpr unsigned int nDSPa = 27;
    //constexpr unsigned int nDSPb = 18;
    constexpr unsigned int nDSPc = 48;
    constexpr unsigned int nDSPd = 48;

    // unit transformations: firmware uses mm, software uses cm
    float BeamWindow = cm_to_mm * beamWindowZ_;
    float T_rphi = cm_to_mm * chosenRofPhi_;
    float T_rz = cm_to_mm * chosenRofZ_;

    // actual algorithm as used in firmware, mostly using same variable names
    float Beam_over_T = BeamWindow / T_rz;
    // Value chosen so that number digitized below when calculating "bot" uses most of the nDSPa bits, without overflowing them. This is done assuming reference number of bits for rT and z mentioned above.
    unsigned int nShiftA = 24;
    // Guess from to keep "bot" in correct range (nDSPa) if number of digitsation bits are changed.
    nShiftA += (rtBits - rtBitsRef) - (zBits - zBitsRef);
    float Beam_over_T_base = 1. / (1 << nShiftA);
    int64_t bot = forceBitWidth(Beam_over_T * rTBase / zBase / Beam_over_T_base, nDSPa);
    int64_t bw = forceBitWidth(BeamWindow / zBase / Beam_over_T_base, nDSPc);
    float etaSecMid = (settings_->etaRegions()[iEtaReg_] + settings_->etaRegions()[iEtaReg_ + 1]) / 2.0;
    float tanlSecMid = 1.0 / tan(2.0 * atan(exp(-etaSecMid)));
    // Value chosen so that number digitized below when calculating "tanlSec_Mid" uses most of the nDSPa bits, without overflowing them. This is done assuming reference number of bits for rT and z mentioned above.
    unsigned int nShiftB = 16;
    // Guess to keep "tanlSec_Mid" in correct range (nDSPa) if number of digitsation bits are changed.
    nShiftB += (rtBits - rtBitsRef) - (zBits - zBitsRef);
    float tanlSecBase = 1. / (1 << nShiftB);
    int64_t tanlSec_Mid = forceBitWidth(int(tanlSecMid * rTBase / zBase / tanlSecBase), nDSPa);
    // Number of extra bits used to digitise r instead of rT within GP code, if both encoded as signed int.
    constexpr unsigned int nExtraBitsR = 2;
    unsigned int rBits = rtBits + nExtraBitsR;
    int64_t r = forceBitWidth(rT + T_rphi / rTBase, rBits);
    int64_t g = forceBitWidth(bot * r - bw, nDSPd);
    int64_t absg = abs(g);
    // Number of useful bits left of the nDSPd assigned to "absg" after right-shifting by nShiftA bits.
    const unsigned nBitsRemainingA = nDSPd - nShiftA;
    int64_t shift_g = forceBitWidth((absg >> nShiftA), nBitsRemainingA);
    // Number of bits is sum of those in two numbers being multiplied.
    int64_t tlsr = forceBitWidth(tanlSec_Mid * r, nDSPa + rBits);
    // Number of useful bits left of (nDSPa + rBits) assigned to "tlsr" after right-shifting by nShiftB bits.
    const unsigned nBitsRemainingB = (nDSPa + rBits) - nShiftB;
    int64_t shift_tlsr = forceBitWidth((tlsr >> nShiftB), nBitsRemainingB);

    vector<bool> insideVec;
    insideVec.push_back(z <= (shift_tlsr + shift_g));
    insideVec.push_back(z >= (shift_tlsr - shift_g));
    return insideVec;
  }

}  // namespace tmtt
