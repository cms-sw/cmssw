#include "L1Trigger/TrackFindingTMTT/interface/ConverterToTTTrack.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

namespace tmtt {

  //=== Convert L1fittedTrack or L1track3D (track candidates after/before fit) to TTTrack format.

  TTTrack<Ref_Phase2TrackerDigi_> ConverterToTTTrack::makeTTTrack(const L1trackBase* trk,
                                                                  unsigned int iPhiSec,
                                                                  unsigned int iEtaReg) const {
    unsigned int nPar, hitPattern;
    double d0, z0, tanL, chi2rphi, chi2rz;

    const L1fittedTrack* fitTrk = dynamic_cast<const L1fittedTrack*>(trk);

    // Handle variables that differ for L1fittedTrack & L1track3D
    if (fitTrk == nullptr) {
      // This is an L1track3D type (track before fit)
      nPar = 4;  // Before fit, TMTT algorithm assumes 4 helix params
      // Set to zero variables that are unavailable for this track type.
      hitPattern = 0;
      d0 = 0.;
      z0 = 0;
      tanL = 0;
      chi2rphi = 0.;
      chi2rz = 0;
    } else {
      // This is an L1fittedTrack type (track after fit)
      if (not fitTrk->accepted())
        throw cms::Exception("LogicError") << "ConverterToTTTrack ERROR: requested to convert invalid L1fittedTrack";
      nPar = fitTrk->nHelixParam();  // Number of helix parameters in track fit
      hitPattern = fitTrk->hitPattern();
      d0 = fitTrk->d0();
      z0 = fitTrk->z0();
      tanL = fitTrk->tanLambda();
      chi2rphi = fitTrk->chi2rphi();
      chi2rz = fitTrk->chi2rz();
    }

    const double& rinv = invPtToInvR_ * trk->qOverPt();
    const double& phi0 = trk->phi0();
    constexpr double mva = -1.;  // MVA quality flags not yet set.
    const double& magneticField = settings_->magneticField();

    TTTrack<Ref_Phase2TrackerDigi_> track(
        rinv, phi0, tanL, z0, d0, chi2rphi, chi2rz, mva, mva, mva, hitPattern, nPar, magneticField);

    // Set references to stubs on this track.
    std::vector<TTStubRef> ttstubrefs = this->stubRefs(trk);
    track.setStubRefs(ttstubrefs);

    // Note which (eta,phi) sector this track was reconstructed in.
    track.setPhiSector(iPhiSec);
    track.setEtaSector(iEtaReg);

    track.setStubPtConsistency(-1);  // not yet filled.

    return track;
  }

  //=== Get references to stubs on track. (Works for either L1track3D or L1fittedTrack).

  std::vector<TTStubRef> ConverterToTTTrack::stubRefs(const L1trackBase* trk) const {
    std::vector<TTStubRef> ttstubrefs;
    const std::vector<Stub*>& stubs = trk->stubs();
    for (Stub* s : stubs) {
      const TTStubRef& ref = s->ttStubRef();
      ttstubrefs.push_back(ref);
    }
    return ttstubrefs;
  }

}  // namespace tmtt
