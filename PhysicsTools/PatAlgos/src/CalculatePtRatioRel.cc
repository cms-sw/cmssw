#include "PhysicsTools/PatAlgos/interface/CalculatePtRatioRel.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

using namespace pat;

CalculatePtRatioRel::CalculatePtRatioRel(float dR2max) : dR2max_(dR2max) {}

CalculatePtRatioRel::~CalculatePtRatioRel() {}

namespace {

  float ptRel(const reco::Candidate::LorentzVector& muP4,
              const reco::Candidate::LorentzVector& jetP4,
              bool subtractMuon = true) {
    reco::Candidate::LorentzVector jp4 = jetP4;
    if (subtractMuon)
      jp4 -= muP4;
    float dot = muP4.Vect().Dot(jp4.Vect());
    float ptrel = muP4.P2() - dot * dot / jp4.P2();
    ptrel = ptrel > 0 ? sqrt(ptrel) : 0.0;
    return ptrel;
  }
}  // namespace

std::array<float, 2> CalculatePtRatioRel::computePtRatioRel(const pat::Muon& muon,
                                                            const reco::JetTagCollection& bTags,
                                                            const reco::JetCorrector* correctorL1,
                                                            const reco::JetCorrector* correctorL1L2L3Res) const {
  //Initialise loop variables
  float minDr2 = 9999;
  double jecL1L2L3Res = 1.;
  double jecL1 = 1.;

  // Compute corrected isolation variables
  const float chIso = muon.pfIsolationR04().sumChargedHadronPt;
  const float nIso = muon.pfIsolationR04().sumNeutralHadronEt;
  const float phoIso = muon.pfIsolationR04().sumPhotonEt;
  const float puIso = muon.pfIsolationR04().sumPUPt;
  const float dbCorrectedIsolation = chIso + std::max(nIso + phoIso - .5f * puIso, 0.f);
  const float dbCorrectedRelIso = dbCorrectedIsolation / muon.pt();

  float JetPtRatio = 1. / (1 + dbCorrectedRelIso);
  float JetPtRel = 0.;

  const reco::Candidate::LorentzVector& muP4(muon.p4());
  for (const auto& tagI : bTags) {
    // for each muon with the lepton
    float dr2 = deltaR2(*(tagI.first), muon);
    if (dr2 > minDr2)
      continue;
    minDr2 = dr2;

    reco::Candidate::LorentzVector jetP4(tagI.first->p4());

    if (correctorL1 && correctorL1L2L3Res) {
      jecL1L2L3Res = correctorL1L2L3Res->correction(*(tagI.first));
      jecL1 = correctorL1->correction(*(tagI.first));
    }

    if (minDr2 < dR2max_) {
      if ((jetP4 - muP4).Rho() < 0.0001) {
        JetPtRel = 0.;
        JetPtRatio = 1.;
      } else {
        jetP4 -= muP4 / jecL1;
        jetP4 *= jecL1L2L3Res;
        jetP4 += muP4;

        JetPtRatio = muP4.pt() / jetP4.pt();
        JetPtRel = ptRel(muP4, jetP4);
      }
    }
  }

  if (JetPtRatio > 1.5)
    JetPtRatio = 1.5;

  std::array<float, 2> outputs = {{JetPtRatio, JetPtRel}};
  return outputs;
};
