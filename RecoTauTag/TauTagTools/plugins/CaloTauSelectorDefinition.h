#ifndef RecoTauTag_TauTagTools_CaloTauSelectorDefinition
#define RecoTauTag_TauTagTools_CaloTauSelectorDefinition

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <iostream>

struct CaloTauSelectorDefinition {

  typedef reco::CaloTauCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::CaloTau *> container;
  typedef container::const_iterator const_iterator;

  struct TDiscCutPair {
    edm::Handle<reco::CaloTauDiscriminator> m_discHandle;
    double m_cut;
  };
  typedef std::vector<TDiscCutPair> TDiscCutPairVec;

  CaloTauSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) {
    discriminatorsPSets_ = cfg.getParameter< std::vector<edm::ParameterSet> >( "discriminators" );
    discriminatorTokens_ = edm::vector_transform( discriminatorsPSets_, [&](edm::ParameterSet const & pSet){return iC.consumes<reco::CaloTauDiscriminator>(pSet.getParameter<edm::InputTag>("discriminator"));} );
    cut_ = ( cfg.exists("cut") ) ? new StringCutObjectSelector<reco::CaloTau>( cfg.getParameter<std::string>( "cut" ) ) : 0;
  }

  ~CaloTauSelectorDefinition () { delete cut_; }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const HandleToCollection & hc,
	       const edm::Event & e,
	       const edm::EventSetup& s)
  {

    selected_.clear();
    assert( hc.isValid() ); // collection of CaloTaus

    // get discriminators and coresponding cuts from stored vpset
    discriminators_.clear();

    std::vector< edm::EDGetTokenT<reco::CaloTauDiscriminator> >::iterator itToken = discriminatorTokens_.begin();
    for(std::vector< edm::ParameterSet >::iterator it = discriminatorsPSets_.begin();
        it != discriminatorsPSets_.end();
        ++it, ++itToken)
    {
      TDiscCutPair disc;
      // get discriminator, check if valid
      // assert isn't a good method to do it
      if(!(e.getByToken( *itToken, disc.m_discHandle))) assert(0);
      disc.m_cut = it->getParameter<double>("selectionCut");
      // the discriminator collection and the CaloTau collection
      // must have the same size
      assert( disc.m_discHandle->size() ==  hc->size());
      discriminators_.push_back(disc);
    }

    unsigned key=0;
    bool passedAllCuts;
    for( collection::const_iterator calotau = hc->begin();
          calotau != hc->end();
          ++calotau, ++key)
    {
      passedAllCuts = true;
      reco::CaloTauRef calotauRef(hc, key);

      //iterate over all discriminators, check the cuts
      for (TDiscCutPairVec::iterator discIt = discriminators_.begin();
           discIt!=discriminators_.end();
           ++discIt)
      {
        if ( (*(discIt->m_discHandle))[calotauRef] <= discIt->m_cut)
          passedAllCuts = false;
      }

      if ( cut_ ) passedAllCuts &= (*cut_)(*calotau);

      if(passedAllCuts)
        selected_.push_back( new reco::CaloTau(*calotau) );
    } // end collection iteration
  } // end select()

  size_t size() const { return selected_.size(); }

 private:
  TDiscCutPairVec discriminators_;
  container selected_;
  std::vector< edm::ParameterSet > discriminatorsPSets_;
  std::vector< edm::EDGetTokenT<reco::CaloTauDiscriminator> > discriminatorTokens_;
  StringCutObjectSelector<reco::CaloTau>* cut_;

};

#endif
