#include "DQM/Physics/interface/TopDQMHelpers.h"

Calculate::Calculate(int maxNJets, double wMass)
    : failed_(false),
      maxNJets_(maxNJets),
      wMass_(wMass),
      massWBoson_(-1.),
      massTopQuark_(-1.),
      massBTopQuark_(-1.),
      tmassWBoson_(-1),
      tmassTopQuark_(-1) {}

double Calculate::massWBoson(const std::vector<reco::Jet>& jets) {
  if (!failed_ && massWBoson_ < 0) operator()(jets);
  return massWBoson_;
}

double Calculate::massTopQuark(const std::vector<reco::Jet>& jets) {
  if (!failed_ && massTopQuark_ < 0) operator()(jets);
  return massTopQuark_;
}

double Calculate::massBTopQuark(const std::vector<reco::Jet>& jets,
                                std::vector<double> VbtagWP, double btagWP_) {
  if (!failed_ && massBTopQuark_ < 0) operator2(jets, VbtagWP, btagWP_);
  return massBTopQuark_;
}

double Calculate::tmassWBoson(reco::RecoCandidate* mu, const reco::MET& met,
                              const reco::Jet& b) {
  if (tmassWBoson_ < 0) operator()(b, mu, met);
  return tmassWBoson_;
}

double Calculate::tmassTopQuark(reco::RecoCandidate* lepton,
                                const reco::MET& met, const reco::Jet& b) {
  if (tmassTopQuark_ < 0) operator()(b, lepton, met);
  return tmassTopQuark_;
}

void Calculate::operator()(const reco::Jet& bJet, reco::RecoCandidate* lepton,
                           const reco::MET& met) {
  double metT = sqrt(pow(met.px(), 2) + pow(met.py(), 2));
  double lepT = sqrt(pow(lepton->px(), 2) + pow(lepton->py(), 2));
  double bT = sqrt(pow(bJet.px(), 2) + pow(bJet.py(), 2));
  reco::Particle::LorentzVector WT = lepton->p4() + met.p4();
  tmassWBoson_ =
      sqrt(pow(metT + lepT, 2) - (WT.px() * WT.px()) - (WT.py() * WT.py()));
  reco::Particle::LorentzVector topT = WT + bJet.p4();
  tmassTopQuark_ = sqrt(pow((metT + lepT + bT), 2) - (topT.px() * topT.px()) -
                        (topT.py() * topT.py()));
}

void Calculate::operator()(const std::vector<reco::Jet>& jets) {
  if (maxNJets_ < 0) maxNJets_ = jets.size();
  failed_ = jets.size() < (unsigned int)maxNJets_;
  if (failed_) {
    return;
  }

  // associate those jets with maximum pt of the vectorial
  // sum to the hadronic decay chain
  double maxPt = -1.;
  std::vector<int> maxPtIndices;
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);

  for (int idx = 0; idx < maxNJets_; ++idx) {
    for (int jdx = 0; jdx < maxNJets_; ++jdx) {
      if (jdx <= idx) continue;
      for (int kdx = 0; kdx < maxNJets_; ++kdx) {
        if (kdx == idx || kdx == jdx) continue;
        reco::Particle::LorentzVector sum =
            jets[idx].p4() + jets[jdx].p4() + jets[kdx].p4();
        if (maxPt < 0. || maxPt < sum.pt()) {
          maxPt = sum.pt();
          maxPtIndices.clear();
          maxPtIndices.push_back(idx);
          maxPtIndices.push_back(jdx);
          maxPtIndices.push_back(kdx);
        }
      }
    }
  }
  massTopQuark_ = (jets[maxPtIndices[0]].p4() + jets[maxPtIndices[1]].p4() +
                   jets[maxPtIndices[2]].p4()).mass();

  // associate those jets that get closest to the W mass
  // with their invariant mass to the W boson
  double wDist = -1.;
  std::vector<int> wMassIndices;
  wMassIndices.push_back(-1);
  wMassIndices.push_back(-1);
  for (unsigned idx = 0; idx < maxPtIndices.size(); ++idx) {
    for (unsigned jdx = 0; jdx < maxPtIndices.size(); ++jdx) {
      if (jdx == idx || maxPtIndices[idx] > maxPtIndices[jdx]) continue;
      reco::Particle::LorentzVector sum =
          jets[maxPtIndices[idx]].p4() + jets[maxPtIndices[jdx]].p4();
      if (wDist < 0. || wDist > fabs(sum.mass() - wMass_)) {
        wDist = fabs(sum.mass() - wMass_);
        wMassIndices.clear();
        wMassIndices.push_back(maxPtIndices[idx]);
        wMassIndices.push_back(maxPtIndices[jdx]);
      }
    }
  }
  massWBoson_ =
      (jets[wMassIndices[0]].p4() + jets[wMassIndices[1]].p4()).mass();
}

void Calculate::operator2(const std::vector<reco::Jet>& jets,
                          std::vector<double> bjet, double btagWP) {
  if (maxNJets_ < 0) maxNJets_ = jets.size();
  failed_ = jets.size() < (unsigned int)maxNJets_;
  if (failed_) {
    return;
  }
  if (jets.size() != bjet.size()) {
    return;
  }

  // associate those jets with maximum pt of the vectorial
  // sum to the hadronic decay chain. Require ONLY 1 btagged jet
  double maxBPt = -1.;
  std::vector<int> maxBPtIndices;
  maxBPtIndices.push_back(-1);
  maxBPtIndices.push_back(-1);
  maxBPtIndices.push_back(-1);
  for (int idx = 0; idx < maxNJets_; ++idx) {
    for (int jdx = 0; jdx < maxNJets_; ++jdx) {
      if (jdx <= idx) continue;
      for (int kdx = 0; kdx < maxNJets_; ++kdx) {
        if (kdx == idx || kdx == jdx) continue;
        // require only 1b-jet
        if ((bjet[idx] > btagWP && bjet[jdx] <= btagWP &&
             bjet[kdx] <= btagWP) ||
            (bjet[idx] <= btagWP && bjet[jdx] > btagWP &&
             bjet[kdx] <= btagWP) ||
            (bjet[idx] <= btagWP && bjet[jdx] <= btagWP &&
             bjet[kdx] > btagWP)) {
          reco::Particle::LorentzVector sum =
              jets[idx].p4() + jets[jdx].p4() + jets[kdx].p4();
          if (maxBPt < 0. || maxBPt < sum.pt()) {
            maxBPt = sum.pt();
            maxBPtIndices.clear();
            maxBPtIndices.push_back(idx);
            maxBPtIndices.push_back(jdx);
            maxBPtIndices.push_back(kdx);
          }
        }
      }
    }
  }
  if (maxBPtIndices[0] < 0 || maxBPtIndices[1] < 0 || maxBPtIndices[2] < 0)
    return;
  massBTopQuark_ = (jets[maxBPtIndices[0]].p4() + jets[maxBPtIndices[1]].p4() +
                    jets[maxBPtIndices[2]].p4()).mass();
}
