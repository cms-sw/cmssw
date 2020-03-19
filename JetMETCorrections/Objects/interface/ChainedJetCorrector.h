//
// Original Author:  Fedor Ratnikov Dec 27, 2006
//
// Correction which chains other corrections
//
#ifndef ChainedJetCorrector_h
#define ChainedJetCorrector_h

#include <vector>

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

class ChainedJetCorrector : public JetCorrector {
public:
  ChainedJetCorrector() {}
  ~ChainedJetCorrector() override {}

  double correction(const JetCorrector::LorentzVector& fJet) const override;
  double correction(const reco::Jet& fJet) const override;
  double correction(const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const override;
  double correction(const reco::Jet& fJet,
                    const edm::RefToBase<reco::Jet>& fJetRef,
                    const edm::Event& fEvent,
                    const edm::EventSetup& fSetup) const override;

  bool eventRequired() const override;
  bool refRequired() const override;

  void push_back(const JetCorrector* fCorrector) { mCorrectors.push_back(fCorrector); }
  void clear() { mCorrectors.clear(); }

private:
  std::vector<const JetCorrector*> mCorrectors;
};

#endif
