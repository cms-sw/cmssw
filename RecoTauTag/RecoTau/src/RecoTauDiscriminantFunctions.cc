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

/*
 * HPStanc variables
 */

double JetPt(Tau tau) {
  return tau.jetRef()->pt();
}

double JetEta(Tau tau) {
  return tau.jetRef()->eta();
}

double JetWidth(Tau tau) {
  return std::sqrt(
      std::abs(tau.jetRef()->etaetaMoment()) +
      std::abs(tau.jetRef()->phiphiMoment()));
}

double SignalPtFraction(Tau tau) {
  return tau.pt()/tau.jetRef()->pt();
}

double IsolationChargedPtFraction(Tau tau) {
  return tau.isolationPFChargedHadrCandsPtSum()/tau.jetRef()->pt();
}

double IsolationECALPtFraction(Tau tau) {
  return tau.isolationPFGammaCandsEtSum()/tau.jetRef()->pt();
}

double IsolationNeutralHadronPtFraction(Tau tau) {
  double sum = 0.0;
  BOOST_FOREACH(PFCandidateRef cand, tau.isolationPFNeutrHadrCands()) {
    sum += cand->pt();
  }
  return sum/tau.jetRef()->pt();
}

double ScaledEtaJetCollimation(Tau tau) {
  return tau.jetRef()->pt()*sqrt(std::abs(
          tau.jetRef()->etaetaMoment()));
}

double ScaledOpeningDeltaR(Tau tau) {
  double max = 0.0;
  const PFCandidateRefVector& cands = tau.signalPFCands();
  for (size_t i = 0; i < cands.size()-1; ++i) {
    for (size_t j = i+1; j < cands.size(); ++j) {
      double deltaRVal = deltaR(cands[i]->p4(), cands[j]->p4());
      if (deltaRVal > max) {
        max = deltaRVal;
      }
    }
  }
  // Correct for resolution
  if ( max < 0.05 )
    max = 0.05;
  // Make invariant of PT
  return max*tau.pt();;
}

double ScaledPhiJetCollimation(Tau tau) {
  return tau.jetRef()->pt()*sqrt(std::abs(
          tau.jetRef()->phiphiMoment()));
}

double IsolationChargedAveragePtFraction(Tau tau) {
  size_t nIsoCharged = tau.isolationPFChargedHadrCands().size();
  double averagePt = (nIsoCharged) ?
      tau.isolationPFChargedHadrCandsPtSum()/nIsoCharged : 0;
  return averagePt/tau.leadPFChargedHadrCand()->pt();
}

double MainTrackPtFraction(Tau tau) {
  return mainTrack(tau)->pt()/tau.jetRef()->pt();
}

VDouble Dalitz2(Tau tau) {
  PFCandidateRef theMainTrack = mainTrack(tau);
  PFCandidateRefVector otherSignalTracks = notMainTrack(tau);
  const std::vector<RecoTauPiZero> &pizeros = tau.signalPiZeroCandidates();
  VDouble output;
  output.reserve(otherSignalTracks.size() + pizeros.size());
  // Add combos with tracks
  BOOST_FOREACH(PFCandidateRef trk, otherSignalTracks) {
    reco::Candidate::LorentzVector p4 = theMainTrack->p4() + trk->p4();
    output.push_back(p4.mass());
  }
  // Add combos with pizeros
  BOOST_FOREACH(const RecoTauPiZero &pizero, pizeros) {
    reco::Candidate::LorentzVector p4 = theMainTrack->p4() + pizero.p4();
    output.push_back(p4.mass());
  }
  return output;
}

double IsolationChargedSumHard(Tau tau) {
  VDouble isocands = extract(tau.isolationPFChargedHadrCands(),
                             std::mem_fun_ref(&PFCandidate::pt));
  double output = 0.0;
  BOOST_FOREACH(double pt, isocands) {
    if (pt > 1.0)
      output += pt;
  }
  return output;
}

double IsolationChargedSumSoft(Tau tau) {
  VDouble isocands = extract(tau.isolationPFChargedHadrCands(),
                             std::mem_fun_ref(&PFCandidate::pt));
  double output = 0.0;
  BOOST_FOREACH(double pt, isocands) {
    if (pt < 1.0)
      output += pt;
  }
  return output;
}

// Relative versions.
double IsolationChargedSumHardRelative(Tau tau) {
  return IsolationChargedSumHard(tau)/tau.jetRef()->pt();
}

double IsolationChargedSumSoftRelative(Tau tau) {
  return IsolationChargedSumSoft(tau)/tau.jetRef()->pt();
}

double IsolationECALSumHard(Tau tau) {
  VDouble isocands = extract(tau.isolationPFGammaCands(),
                             std::mem_fun_ref(&PFCandidate::pt));
  double output = 0.0;
  BOOST_FOREACH(double pt, isocands) {
    if (pt > 1.5)
      output += pt;
  }
  return output;
}

double IsolationECALSumSoft(Tau tau) {
  VDouble isocands = extract(tau.isolationPFGammaCands(),
                             std::mem_fun_ref(&PFCandidate::pt));
  double output = 0.0;
  BOOST_FOREACH(double pt, isocands) {
    if (pt < 1.5)
      output += pt;
  }
  return output;
}

// Relative versions.
double IsolationECALSumHardRelative(Tau tau) {
  return IsolationECALSumHard(tau)/tau.jetRef()->pt();
}
double IsolationECALSumSoftRelative(Tau tau) {
  return IsolationECALSumSoft(tau)/tau.jetRef()->pt();
}

double EMFraction(Tau tau) {
  return tau.emFraction();
}

double ImpactParameterSignificance(Tau tau) {
  return std::abs(tau.leadPFChargedHadrCandsignedSipt());
}

double OutlierN(Tau tau) {
  return tau.isolationPFChargedHadrCands().size() +
      tau.isolationPFGammaCands().size();
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
  return Dalitz2(tau);
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

