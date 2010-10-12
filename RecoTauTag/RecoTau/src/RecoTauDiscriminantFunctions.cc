#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantFunctions.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <algorithm>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include <boost/foreach.hpp>

namespace reco { namespace tau { namespace disc {

// Helper functions
namespace {

const PFCandidate& removeRef(const PFCandidateRef& pfRef) {
  return *pfRef;
}

const RecoTauPiZero& removeRef(const RecoTauPiZero& piZero) {
  return piZero;
}

// A PFTau member function
template<typename Collection, typename Function>
VDouble extract(const Collection& cands, Function func) {
  // #define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
  VDouble output;
  output.reserve(cands.size());
  for(typename Collection::const_iterator cand = cands.begin();
      cand != cands.end(); ++cand) {
    output.push_back(func(removeRef(*cand)));
  }
  return output;
}

class DeltaRToAxis {
  public:
    DeltaRToAxis(const reco::Candidate::LorentzVector& axis):axis_(axis){}
    double operator()(const Candidate& cand)
    {
      return deltaR(cand.p4(), axis_);
    }
  private:
    const reco::Candidate::LorentzVector& axis_;
};

} // end helper functions

PFCandidateRef mainTrack(Tau tau) {
  if (tau.signalPFChargedHadrCands().size() ==  3) {
    for (size_t itrk = 0; itrk < 3; ++itrk) {
      if (tau.signalPFChargedHadrCands()[itrk]->charge() * tau.charge() < 0)
        return tau.signalPFChargedHadrCands()[itrk];
    }
  }
  return tau.leadPFChargedHadrCand();
}

PFCandidateRefVector notMainTrack(Tau tau)
{
  const PFCandidateRef& mainTrackRef = mainTrack(tau);
  PFCandidateRefVector output;
  output.reserve(tau.signalPFChargedHadrCands().size() - 1);
  BOOST_FOREACH(const PFCandidateRef& ref, tau.signalPFChargedHadrCands()) {
    if (ref != mainTrackRef)
      output.push_back(ref);
  }
  return output;
}

double OutlierN(Tau tau) {
  return tau.isolationPFChargedHadrCands().size() +
      tau.isolationPiZeroCandidates().size();
}

double OutlierNCharged(Tau tau) {
  return tau.isolationPFChargedHadrCands().size();
}

double MainTrackPt(Tau tau) {
  PFCandidateRef trk = mainTrack(tau);
  return (!trk) ? 0.0 : trk->pt();
}

double MainTrackEta(Tau tau) {
  PFCandidateRef trk = mainTrack(tau);
  return (!trk) ? 0.0 : trk->eta();
}

double MainTrackAngle(Tau tau) {
  PFCandidateRef trk = mainTrack(tau);
  return (!trk) ? 0.0 : deltaR(trk->p4(), tau.p4());
}

double OutlierSumPt(Tau tau) {
  return tau.isolationPFChargedHadrCandsPtSum() +
      tau.isolationPFGammaCandsEtSum();
}

double ChargedOutlierSumPt(Tau tau) {
  return tau.isolationPFChargedHadrCandsPtSum();
}

double NeutralOutlierSumPt(Tau tau) {
  return tau.isolationPFGammaCandsEtSum();
}

// Quantities associated to tracks - that are not the main track
VDouble TrackPt(Tau tau) {
  return extract(notMainTrack(tau), std::mem_fun_ref(&PFCandidate::pt));
}

VDouble TrackEta(Tau tau) {
  return extract(notMainTrack(tau), std::mem_fun_ref(&PFCandidate::eta));
}

VDouble TrackAngle(Tau tau) {
  return extract(notMainTrack(tau), DeltaRToAxis(tau.p4()));
}

// Quantities associated to PiZeros
VDouble PiZeroPt(Tau tau) {
  return extract(tau.signalPiZeroCandidates(), std::mem_fun_ref(&RecoTauPiZero::pt));
}

VDouble PiZeroEta(Tau tau) {
  return extract(tau.signalPiZeroCandidates(), std::mem_fun_ref(&RecoTauPiZero::eta));
}

VDouble PiZeroAngle(Tau tau) {
  return extract(tau.signalPiZeroCandidates(), DeltaRToAxis(tau.p4()));
}

// Isolation quantities
VDouble OutlierPt(Tau tau) {
  return extract(tau.isolationPFCands(), std::mem_fun_ref(&PFCandidate::pt));
}

VDouble OutlierAngle(Tau tau) {
  return extract(tau.isolationPFCands(), DeltaRToAxis(tau.p4()));
}

VDouble ChargedOutlierPt(Tau tau) {
  return extract(tau.isolationPFChargedHadrCands(),
                 std::mem_fun_ref(&PFCandidate::pt));
}

VDouble ChargedOutlierAngle(Tau tau) {
  return extract(tau.isolationPFChargedHadrCands(), DeltaRToAxis(tau.p4()));
}

VDouble NeutralOutlierPt(Tau tau) {
  return extract(tau.isolationPFGammaCands(),
                 std::mem_fun_ref(&PFCandidate::pt));
}

VDouble NeutralOutlierAngle(Tau tau) {
  return extract(tau.isolationPFGammaCands(), DeltaRToAxis(tau.p4()));
}

// Invariant mass of main track with other combinations
VDouble Dalitz(Tau tau) {
  VDouble output;
  PFCandidateRef main = mainTrack(tau);
  if(main.isNonnull()) {
    output.reserve(tau.signalPFCands().size() - 1);
    for(PFCandidateRefVector::const_iterator signalCand =
        tau.signalPFCands().begin();
        signalCand != tau.signalPFCands().end(); ++signalCand) {
      if(*signalCand != main) {
        reco::Candidate::LorentzVector system = main->p4()+(*signalCand)->p4();
        output.push_back(system.mass());
      }
    }
  }
  return output;
}

// The below functions are deprecated.
// Not used, for backwards compatability
VDouble FilteredObjectPt(Tau tau) { return VDouble(); }
VDouble GammaOccupancy(Tau tau) { return VDouble(); }
VDouble GammaPt(Tau tau) { return VDouble(); }
VDouble InvariantMassOfSignalWithFiltered(Tau tau) { return VDouble(); }
VDouble InvariantMass(Tau tau) { return VDouble(); }
VDouble OutlierMass(Tau tau) { return VDouble(); }

}}} // end reco::tau::disc namespace

