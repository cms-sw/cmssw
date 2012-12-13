#ifndef RecoTauTag_TauTagTools_PFTauSelectorDefinition
#define RecoTauTag_TauTagTools_PFTauSelectorDefinition

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <boost/foreach.hpp>

#include <TFormula.h>

#include <iostream>

struct PFTauSelectorDefinition {

  typedef reco::PFTauCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< const reco::PFTau *> container;
  typedef container::const_iterator const_iterator;

  struct DiscCutPair 
  {
    DiscCutPair() 
      : cutFormula_(0)
    {}
    ~DiscCutPair() 
    {
      delete cutFormula_;
    }
    edm::Handle<reco::PFTauDiscriminator> handle_;
    edm::InputTag inputTag_;
    double cut_;
    TFormula* cutFormula_;
  };
  typedef std::vector<DiscCutPair*> DiscCutPairVec;

  PFTauSelectorDefinition (const edm::ParameterSet &cfg) {
    std::vector<edm::ParameterSet> discriminators =
      cfg.getParameter<std::vector<edm::ParameterSet> >("discriminators");
    // Build each of our cuts
    BOOST_FOREACH(const edm::ParameterSet& pset, discriminators) {
      DiscCutPair* newCut = new DiscCutPair();
      newCut->inputTag_ = pset.getParameter<edm::InputTag>("discriminator");
      if ( pset.existsAs<std::string>("selectionCut") ) newCut->cutFormula_ = new TFormula("selectionCut", pset.getParameter<std::string>("selectionCut").data());
      else newCut->cut_ = pset.getParameter<double>("selectionCut");
      discriminators_.push_back(newCut);
    }

    // Build a string cut if desired
    if (cfg.exists("cut")) {
      cut_.reset(new StringCutObjectSelector<reco::PFTau>(
            cfg.getParameter<std::string>( "cut" )));
    }
  }

  ~PFTauSelectorDefinition()
  {
    BOOST_FOREACH(DiscCutPair* disc, discriminators_) {
      delete disc;
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
    BOOST_FOREACH(DiscCutPair* disc, discriminators_) {
      e.getByLabel(disc->inputTag_, disc->handle_);
    }

    const size_t nTaus = hc->size();
    for (size_t iTau = 0; iTau < nTaus; ++iTau) {
      bool passed = true;
      reco::PFTauRef tau(hc, iTau);
      //std::cout << "PFTauSelector: Pt = " << tau->pt() << ", eta = " << tau->eta() << ", phi = " << tau->phi() << std::endl;
      // Check if it passed all the discrimiantors
      BOOST_FOREACH(const DiscCutPair* disc, discriminators_) {
        // Check this discriminator passes
	bool passedDisc = true;
	if ( disc->cutFormula_ ) {
	  passedDisc = (disc->cutFormula_->Eval((*disc->handle_)[tau]) > 0.5);
	  //std::cout << "formula = " << disc->cutFormula_->GetTitle() << ", discr = " << (*disc->handle_)[tau] << ": passedDisc = " << passedDisc << std::endl;
	} else {
	  passedDisc = ((*disc->handle_)[tau] > disc->cut_);
	}
        if ( !passedDisc ) {
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
