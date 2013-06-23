#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

class RecoTauDifferenceAnalyzer : public edm::EDFilter {
  public:
    explicit RecoTauDifferenceAnalyzer(const edm::ParameterSet& pset);
    virtual ~RecoTauDifferenceAnalyzer() {}
    virtual bool filter(edm::Event& evt, const edm::EventSetup& es);
    virtual void endJob();
  private:
    edm::InputTag src1_;
    edm::InputTag src2_;
    edm::InputTag disc1_;
    edm::InputTag disc2_;
    double maxDeltaR_;
    size_t eventsExamined_;
    size_t tausExamined_;
    size_t differences_;
    size_t passed1_;
    size_t passed2_;
    size_t allPassed1_;
    size_t allPassed2_;
    bool filter_;
};

RecoTauDifferenceAnalyzer::RecoTauDifferenceAnalyzer(
    const edm::ParameterSet& pset) {
  src1_ = pset.getParameter<edm::InputTag>("src1");
  src2_ = pset.getParameter<edm::InputTag>("src2");
  disc1_ = pset.getParameter<edm::InputTag>("disc1");
  disc2_ = pset.getParameter<edm::InputTag>("disc2");
  eventsExamined_ = 0;
  tausExamined_ = 0;
  differences_ = 0;
  passed1_ = 0;
  passed2_ = 0;
  allPassed2_ = 0;
  allPassed1_ = 0;
  filter_ = pset.exists("filter") ? pset.getParameter<bool>("filter") : false;
}

namespace {
  reco::PFJetRef getJetRef(const reco::PFTau& tau) {
    if (tau.jetRef().isNonnull())
      return tau.jetRef();
    else if (tau.pfTauTagInfoRef()->pfjetRef().isNonnull())
      return tau.pfTauTagInfoRef()->pfjetRef();
    else throw cms::Exception("cant find jet ref");
  }
}

bool RecoTauDifferenceAnalyzer::filter(
    edm::Event& evt, const edm::EventSetup& es) {
  eventsExamined_++;
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

  bool differenceFound = false;
  // Loop over first collection
  for (size_t iTau1 = 0; iTau1 < taus1->size(); ++iTau1) {
    tausExamined_++;
    reco::PFTauRef tau1(taus1, iTau1);
    // Find the best match in the other collection
    reco::PFTauRef bestMatch;
    double bestDeltaR = -1;
    for (size_t iTau2 = 0; iTau2 < taus2->size(); ++iTau2) {
      reco::PFTauRef tau2(taus2, iTau2);
      reco::PFJetRef jet1 = getJetRef(*tau1);
      reco::PFJetRef jet2 = getJetRef(*tau2);
      double deltaRVal = deltaR(jet2->p4(), jet1->p4());
      if (bestMatch.isNull() || deltaRVal < bestDeltaR) {
        bestMatch = tau2;
        bestDeltaR = deltaRVal;
      }
    }
    // See what's up with the discriminators
    bool result1 = ((*disc1)[tau1] > 0.5);
    bool result2 = ((*disc2)[bestMatch] > 0.5);
    allPassed1_ += result1;
    allPassed2_ += result2;
    if (result1 ^ result2) {
      differenceFound = true;
      passed1_ += result1;
      passed2_ += result2;
      differences_++;
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
  return (filter_ ? differenceFound : true);
}

void RecoTauDifferenceAnalyzer::endJob() {
  std::cout <<  " RECO TAU DIFFERENCE SUMMARY: " << std::endl;
  std::cout <<  " Examined " << tausExamined_ << " taus in "
    << eventsExamined_ << " events." << std::endl;
  std::cout << " There were " << differences_ << " differences." << std::endl;
  std::cout << src1_ << "," << disc1_ << " had "
    << allPassed1_ << " total passes and "
    << passed1_ << " exclusive passes." << std::endl;
  std::cout << src2_ << "," << disc2_ << " had "
    << allPassed2_ << " total passes and "
    << passed2_ << " exclusive passes." << std::endl;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDifferenceAnalyzer);
