#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

class RecoTauDifferenceAnalyzer : public edm::EDAnalyzer {
  public:
    explicit RecoTauDifferenceAnalyzer(const edm::ParameterSet& pset);
    virtual ~RecoTauDifferenceAnalyzer() {}
    virtual void analyze(const edm::Event& evt, const edm::EventSetup& es);
  private:
    edm::InputTag src1_;
    edm::InputTag src2_;
    edm::InputTag disc1_;
    edm::InputTag disc2_;
    double maxDeltaR_;
};

RecoTauDifferenceAnalyzer::RecoTauDifferenceAnalyzer(
    const edm::ParameterSet& pset) {
  src1_ = pset.getParameter<edm::InputTag>("src1");
  src2_ = pset.getParameter<edm::InputTag>("src2");
  disc1_ = pset.getParameter<edm::InputTag>("disc1");
  disc2_ = pset.getParameter<edm::InputTag>("disc2");
}

void RecoTauDifferenceAnalyzer::analyze(const edm::Event& evt,
                                        const edm::EventSetup& es) {
  // Get taus
  edm::Handle<reco::PFTauCollection> taus1;
  evt.getByLabel(src1_, taus1);
  edm::Handle<reco::PFTauCollection> taus2;
  evt.getByLabel(src2_, taus2);

  // Get discriminators
  edm::Handle<reco::PFTauDiscriminator> disc1;
  evt.getByLabel(disc1_, disc1);
  edm::Handle<reco::PFTauDiscriminator> disc2;
  evt.getByLabel(disc2_, disc2);

  // Loop over first collection
  for (size_t iTau1 = 0; iTau1 < taus1->size(); ++iTau1) {
    reco::PFTauRef tau1(taus1, iTau1);
    // Find the best match in the other collection
    reco::PFTauRef bestMatch;
    double bestDeltaR = -1;
    for (size_t iTau2 = 0; iTau2 < taus2->size(); ++iTau2) {
      reco::PFTauRef tau2(taus2, iTau2);
      double deltaRVal = deltaR(tau2->p4(), tau1->p4());
      if (bestMatch.isNull() || deltaRVal < bestDeltaR) {
        bestMatch = tau2;
        bestDeltaR = deltaRVal;
      }
    }
    // See what's up with the discriminators
    bool result1 = ((*disc1)[tau1] > 0.5);
    bool result2 = ((*disc2)[bestMatch] > 0.5);
    if (result1 ^ result2) {
      std::cout << "********* RecoTau difference detected! *************"
          << std::endl;
      std::cout << " Tau1 InputTag: " << src1_ << " result: " << result1
          << std::endl;
      std::cout << " Tau2 InputTag: " << src2_ << " result: " << result2
          << std::endl;
      std::cout << "---------       Tau 1                  -------------"
          << std::endl;
      std::cout << *tau1 << std::endl;
      tau1->dump(std::cout);
      std::cout << "---------       Tau 2                  -------------"
          << std::endl;
      std::cout << *bestMatch << std::endl;
      bestMatch->dump(std::cout);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDifferenceAnalyzer);
