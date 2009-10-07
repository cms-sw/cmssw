#ifndef JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h
#define JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <vector>

namespace edm { class ParameterSet; }
namespace reco { class GenJet; }

namespace jpt {

  class SortJetCollectionsByGenJetPt : public edm::EDProducer {
  
  public:
  
    explicit SortJetCollectionsByGenJetPt( const edm::ParameterSet& );
    ~SortJetCollectionsByGenJetPt();
  
    virtual void produce( edm::Event&, const edm::EventSetup& );
  
  private:
  
    edm::InputTag src_;
    edm::InputTag matched_;
    std::vector<edm::InputTag> jets_;
  
  };

}

// -----------------------------------------------------------------------------
//
namespace reco { 
  bool operator== ( const reco::Jet& j1, const reco::Jet& j2 ) {
    reco::Jet::Constituents c1 = j1.getJetConstituents();
    reco::Jet::Constituents c2 = j2.getJetConstituents();
    if ( c1.empty() || c2.empty() || c1.size() != c2.size() ) { return false; }
    reco::Jet::Constituents::const_iterator ii = c1.begin();
    reco::Jet::Constituents::const_iterator jj = c1.end();
    for ( ; ii != jj; ++ii ) {
      reco::Jet::Constituents::const_iterator iii = c2.begin();
      reco::Jet::Constituents::const_iterator jjj = c2.end();
      for ( ; iii != jjj; ++iii ) { if ( *iii == *ii ) { break; } }
      if ( iii == c2.end() ) { return false; }
    }
    return true;
  }
}

// -----------------------------------------------------------------------------
//
namespace reco { 
  bool operator< ( const reco::GenJet& left, const reco::GenJet& right )  {
    return ( fabs ( left.pt() - right.pt() ) > std::numeric_limits<double>::epsilon() ?  
	     left.pt() > right.pt() : //@@ NOTE THAT THIS IS EFFECTIVELY GREATER THAN!!!
	     false ); 
  }
}

#endif // JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h
