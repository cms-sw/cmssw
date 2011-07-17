#ifndef CommonTools_ParticleFlow_PFJetSelectorDefinition
#define CommonTools_ParticleFlow_PFJetSelectorDefinition

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "boost/iterator/transform_iterator.hpp"
#include <functional>

namespace pf2pat {

  class PFJetSelectorDefinition {

  public:
    typedef reco::PFJetCollection collection;
    typedef edm::Handle< collection > HandleToCollection;
    typedef std::vector<reco::PFJet>  container;
    
    struct Pointer : public std::unary_function<reco::PFJet,const reco::PFJet *> { 
      const reco::PFJet * operator()(const reco::PFJet &c) const { return &c; } 
    };
    
    typedef boost::transform_iterator<Pointer,container::const_iterator> const_iterator;
    
    PFJetSelectorDefinition () {}
    
    const_iterator begin() const { return const_iterator(selected_.begin()); }
    
    const_iterator end() const { return const_iterator(selected_.end()); }
    
    size_t size() const { return selected_.size(); }
    
    const container& selected() const {return selected_;}
    
  protected:
    container selected_;
  };
}

#endif
