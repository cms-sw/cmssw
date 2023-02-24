#include "PhysicsTools/PatAlgos/interface/CalculatePtRatioRel.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

using namespace pat;

CalculatePtRatioRel::CalculatePtRatioRel(float dRmax) : dRmax_(dRmax) {
}

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

std::vector<float> CalculatePtRatioRel::computePtRatioRel(const pat::Muon& muon,
                                   const reco::JetTagCollection& bTags,
                                   const reco::JetCorrector* correctorL1,
                                   const reco::JetCorrector* correctorL1L2L3Res) const {

  //Initialise loop variables
  double minDr = 9999;
  double jecL1L2L3Res = 1.;
  double jecL1 = 1.;
  float JetPtRatio = 0.0;
  float JetPtRel = 0.0;
  
  // Compute corrected isolation variables
  double chIso = muon.pfIsolationR04().sumChargedHadronPt;
  double nIso = muon.pfIsolationR04().sumNeutralHadronEt;
  double phoIso = muon.pfIsolationR04().sumPhotonEt;
  double puIso = muon.pfIsolationR04().sumPUPt;
  double dbCorrectedIsolation = chIso + std::max(nIso + phoIso - .5 * puIso, 0.);
  double dbCorrectedRelIso = dbCorrectedIsolation / muon.pt();

  JetPtRatio = 1. / (1 + dbCorrectedRelIso);
  JetPtRel = 0;

  for (const auto& tagI : bTags) {
    // for each muon with the lepton
    double dr = deltaR(*(tagI.first), muon);
    if (dr > minDr)
      continue;
    minDr = dr;

    const reco::Candidate::LorentzVector& muP4(muon.p4());
    reco::Candidate::LorentzVector jetP4(tagI.first->p4());

    if (correctorL1 && correctorL1L2L3Res) {
      jecL1L2L3Res = correctorL1L2L3Res->correction(*(tagI.first));
      jecL1 = correctorL1->correction(*(tagI.first));
    }

    if (minDr < dRmax_) {
      if ((jetP4 - muP4).Rho() < 0.0001) {
        JetPtRel = 0;
        JetPtRatio = 1;
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

  std::vector<float> outputs = {JetPtRatio, JetPtRel};
  return outputs;
};
