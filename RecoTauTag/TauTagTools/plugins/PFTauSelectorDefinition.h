#ifndef RecoTauTag_TauTagTools_PFTauSelectorDefinition
#define RecoTauTag_TauTagTools_PFTauSelectorDefinition

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <iostream>

struct PFTauSelectorDefinition {

  typedef reco::PFTauCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< const reco::PFTau *> container;
  typedef container::const_iterator const_iterator;

  struct DiscCutPair {
    edm::Handle<reco::PFTauDiscriminator> handle;
    edm::EDGetTokenT<reco::PFTauDiscriminator> inputToken;
    double cut;
  };
  typedef std::vector<DiscCutPair> DiscCutPairVec;

  PFTauSelectorDefinition (const edm::ParameterSet &cfg, edm::ConsumesCollector && iC) {
    std::vector<edm::ParameterSet> discriminators =
      cfg.getParameter<std::vector<edm::ParameterSet> >("discriminators");
    // Build each of our cuts
    for(auto const& pset : discriminators) {
      DiscCutPair newCut;
      newCut.inputToken = iC.consumes<reco::PFTauDiscriminator>(pset.getParameter<edm::InputTag>("discriminator"));
      newCut.cut = pset.getParameter<double>("selectionCut");
      discriminators_.push_back(newCut);
    }

    // Build a string cut if desired
    if (cfg.exists("cut")) {
      cut_.reset(new StringCutObjectSelector<reco::PFTau>(
            cfg.getParameter<std::string>( "cut" )));
    }
  }

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }

  void select(const HandleToCollection & hc, const edm::Event & e,
      const edm::EventSetup& s) {
    selected_.clear();

    if (!hc.isValid()) {
      throw cms::Exception("PFTauSelectorBadHandle")
        << "an invalid PFTau handle with ProductID"
        << hc.id() << " passed to PFTauSelector.";
    }

    // Load each discriminator
    for(auto& disc : discriminators_) {
      e.getByToken(disc.inputToken, disc.handle);
    }

    const size_t nTaus = hc->size();
    for (size_t iTau = 0; iTau < nTaus; ++iTau) {
      bool passed = true;
      reco::PFTauRef tau(hc, iTau);
      // Check if it passed all the discrimiantors
      for(auto const& disc : discriminators_) {
        // Check this discriminator passes
        if (!((*disc.handle)[tau] > disc.cut)) {
          passed = false;
          break;
        }
      }

      if (passed && cut_.get()) {
        passed = (*cut_)(*tau);
      }

      if (passed)
        selected_.push_back(tau.get());
    }
  } // end select()

  size_t size() const { return selected_.size(); }

 private:
  container selected_;
  DiscCutPairVec discriminators_;
  std::auto_ptr<StringCutObjectSelector<reco::PFTau> > cut_;

};

#endif
