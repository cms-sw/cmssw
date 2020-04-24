#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantFunctions.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/angle.h"
#include <algorithm>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include <boost/foreach.hpp>

namespace reco { namespace tau { namespace disc {

// Helper functions
namespace {

const PFCandidate& removeRef(const PFCandidatePtr& pfRef) {
  return *pfRef;
}

const RecoTauPiZero& removeRef(const RecoTauPiZero& piZero) {
  return piZero;
}

// Function = A PFTau member function
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

PFCandidatePtr mainTrack(Tau tau) {
  if (tau.signalPFChargedHadrCands().size() ==  3) {
    for (size_t itrk = 0; itrk < 3; ++itrk) {
      if (tau.signalPFChargedHadrCands()[itrk]->charge() * tau.charge() < 0)
        return tau.signalPFChargedHadrCands()[itrk];
    }
  }
  return tau.leadPFChargedHadrCand();
}

std::vector<PFCandidatePtr> notMainTrack(Tau tau)
{
  const PFCandidatePtr& mainTrackPtr = mainTrack(tau);
  std::vector<PFCandidatePtr> output;
  output.reserve(tau.signalPFChargedHadrCands().size() - 1);
  BOOST_FOREACH(const PFCandidatePtr& ptr, tau.signalPFChargedHadrCands()) {
    if (ptr != mainTrackPtr)
      output.push_back(ptr);
  }
  return output;
}

double Pt(Tau tau) { return tau.pt(); }
double Eta(Tau tau) { return tau.eta(); }
double AbsEta(Tau tau) { return std::abs(tau.eta()); }
double Mass(Tau tau) { return tau.mass(); }
double DecayMode(Tau tau) { return tau.decayMode(); }
double InvariantMassOfSignal(Tau tau) { return tau.mass(); }

/*
 * HPStanc variables
 */

double JetPt(Tau tau) {
  return tau.jetRef()->pt();
}

double JetEta(Tau tau) {
  return tau.jetRef()->eta();
}

double AbsJetEta(Tau tau) {
  return std::abs(tau.jetRef()->eta());
}

double JetWidth(Tau tau) {
  return std::sqrt(
      std::abs(tau.jetRef()->etaetaMoment()) +
      std::abs(tau.jetRef()->phiphiMoment()));
}

double JetTauDR(Tau tau) {
  return reco::deltaR(tau.p4(), tau.jetRef()->p4());
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
  BOOST_FOREACH(PFCandidatePtr cand, tau.isolationPFNeutrHadrCands()) {
    sum += cand->pt();
  }
  return sum/tau.jetRef()->pt();
}

double ScaledEtaJetCollimation(Tau tau) {
  return tau.jetRef()->pt()*sqrt(std::abs(
          tau.jetRef()->etaetaMoment()));
}

double OpeningDeltaR(Tau tau) {
  double sumEt = 0;
  double weightedDeltaR = 0;
  BOOST_FOREACH(const reco::PFCandidatePtr& cand, tau.signalPFCands()) {
    double candEt = cand->et();
    double candDeltaR = reco::deltaR(cand->p4(), tau.p4());
    sumEt += candEt;
    weightedDeltaR += candDeltaR*candEt;
  }
  return (sumEt > 0) ? weightedDeltaR/sumEt : 0.0;
}

double OpeningAngle3D(Tau tau) {
  double sumE = 0;
  double weightedAngle = 0;
  BOOST_FOREACH(const reco::PFCandidatePtr& cand, tau.signalPFCands()) {
    double candE = cand->energy();
    double candAngle = angle(cand->p4(), tau.p4());
    sumE += candE;
    weightedAngle += candAngle*candE;
  }
  return (sumE > 0) ? weightedAngle/sumE : 0.0;
}

double ScaledOpeningDeltaR(Tau tau) {
  double max = 0.0;
  const std::vector<PFCandidatePtr>& cands = tau.signalPFCands();
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
  PFCandidatePtr theMainTrack = mainTrack(tau);
  std::vector<PFCandidatePtr> otherSignalTracks = notMainTrack(tau);
  const std::vector<RecoTauPiZero> &pizeros = tau.signalPiZeroCandidates();
  VDouble output;
  output.reserve(otherSignalTracks.size() + pizeros.size());
  // Add combos with tracks
  BOOST_FOREACH(PFCandidatePtr trk, otherSignalTracks) {
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
  //double result = tau.emFraction();
  reco::Candidate::LorentzVector gammaP4;
  BOOST_FOREACH(const reco::PFCandidatePtr& gamma, tau.signalPFGammaCands()) {
    gammaP4 += gamma->p4();
  }
  double result = gammaP4.pt()/tau.pt();

  if (result > 0.99) {
    LogDebug("TauDiscFunctions") << "EM fraction = " << result
      << tau ;
      LogDebug("TauDiscFunctions") << "charged" ;
    BOOST_FOREACH(const reco::PFCandidatePtr cand, tau.signalPFChargedHadrCands()) {
      LogDebug("TauDiscFunctions") << " pt: " << cand->pt() << " type: " << cand->particleId() <<  " key: " << cand.key() ;
    }
    LogDebug("TauDiscFunctions") << "gammas" ;
    BOOST_FOREACH(const reco::PFCandidatePtr cand, tau.signalPFGammaCands()) {
      LogDebug("TauDiscFunctions") << " pt: " << cand->pt() << " type: " << cand->particleId() <<  " key: " << cand.key() ;
    }
  }
  return result;
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
  PFCandidatePtr trk = mainTrack(tau);
  return (!trk) ? 0.0 : trk->pt();
}

double MainTrackEta(Tau tau) {
  PFCandidatePtr trk = mainTrack(tau);
  return (!trk) ? 0.0 : trk->eta();
}

double MainTrackAngle(Tau tau) {
  PFCandidatePtr trk = mainTrack(tau);
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

