#ifndef RecoTauTag_TauTagTools_PFTauSelectorDefinition
#define RecoTauTag_TauTagTools_PFTauSelectorDefinition

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <iostream>

struct PFTauSelectorDefinition {

  typedef reco::PFTauCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::PFTau *> container;
  typedef container::const_iterator const_iterator;

  struct TDiscCutPair {
    edm::Handle<reco::PFTauDiscriminator> m_discHandle;
    double m_cut;
  };
  typedef std::vector<TDiscCutPair> TDiscCutPairVec;

  PFTauSelectorDefinition ( const edm::ParameterSet & cfg ) {
    discriminators_ = cfg.getParameter< std::vector<edm::ParameterSet> >( "discriminators" );
    cut_ = ( cfg.exists("cut") ) ? new StringCutObjectSelector<reco::PFTau>( cfg.getParameter<std::string>( "cut" ) ) : 0;
  }

  ~PFTauSelectorDefinition () { delete cut_; }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const HandleToCollection & hc,
	       const edm::Event & e,
	       const edm::EventSetup& s)
  {

    selected_.clear();
    assert( hc.isValid() ); // collection of PFTaus

    // get discriminators and coresponding cuts from stored vpset
    static TDiscCutPairVec discriminators;
    discriminators.clear();

    for(std::vector< edm::ParameterSet >::iterator it = discriminators_.begin();
        it != discriminators_.end();
        ++it)
    {
      TDiscCutPair disc;
      // get discriminator, check if valid
      // assert isn't a good method to do it
      if(!(e.getByLabel( it->getParameter<edm::InputTag>("discriminator"), disc.m_discHandle))) assert(0);
      disc.m_cut = it->getParameter<double>("selectionCut");
      // the discriminator collection and the PFTau collection
      // must have the same size
      assert( disc.m_discHandle->size() ==  hc->size());
      discriminators.push_back(disc);
    }

    unsigned key=0;
    static bool passedAllCuts;
    for( collection::const_iterator pftau = hc->begin();
          pftau != hc->end();
          ++pftau, ++key)
    {
      passedAllCuts = true;
      reco::PFTauRef pfTauRef(hc, key);

      //iterate over all discriminators, check the cuts
      for (TDiscCutPairVec::iterator discIt = discriminators.begin();
           discIt!=discriminators.end();
           ++discIt)
      {
        if ( (*(discIt->m_discHandle))[pfTauRef] <= discIt->m_cut)
          passedAllCuts = false;
      }

      if ( cut_ ) passedAllCuts &= (*cut_)(*pftau);

      if(passedAllCuts)
        selected_.push_back( new reco::PFTau(*pftau) );
    } // end collection iteration
  } // end select()

  size_t size() const { return selected_.size(); }

 private:
  container selected_;
  std::vector< edm::ParameterSet > discriminators_;
  StringCutObjectSelector<reco::PFTau>* cut_;

};

#endif
