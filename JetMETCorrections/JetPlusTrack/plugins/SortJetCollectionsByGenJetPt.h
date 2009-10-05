#ifndef JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h
#define JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <vector>

namespace pat { class Jet; }
namespace edm { class ParameterSet; }

namespace jpt {

  class SortJetCollectionsByGenJetPt : public edm::EDProducer {
  
  public:
  
    explicit SortJetCollectionsByGenJetPt( const edm::ParameterSet& );
    ~SortJetCollectionsByGenJetPt();
  
    virtual void produce( edm::Event&, const edm::EventSetup& );
  
  private:
  
    //edm::InputTag matchedJets_;
    edm::InputTag genJets_;
    std::vector<edm::InputTag> caloJets_;
    std::vector<edm::InputTag> patJets_;
  
  };

}

// -----------------------------------------------------------------------------
//
namespace reco { 
  bool operator== ( const reco::CaloJet& jet1, const reco::CaloJet& jet2 ) {
    std::vector<CaloTowerDetId > towers1 = jet1.getTowerIndices();
    std::vector<CaloTowerDetId > towers2 = jet2.getTowerIndices();
    if ( towers1.empty() || 
	 towers2.empty() || 
	 towers1.size() != towers2.size() ) { return false; }
    std::vector<CaloTowerDetId>::const_iterator ii = towers1.begin();
    std::vector<CaloTowerDetId>::const_iterator jj = towers1.end();
    for ( ; ii != jj; ++ii ) {
      std::vector<CaloTowerDetId>::const_iterator iii = towers2.begin();
      std::vector<CaloTowerDetId>::const_iterator jjj = towers2.end();
      for ( ; iii != jjj; ++iii ) { if ( iii->rawId() == ii->rawId() ) { break; } }
      if ( iii == towers2.end() ) { return false; }
    }
    return true;
  }
}

// -----------------------------------------------------------------------------
//
namespace pat { 
  bool operator== ( const pat::Jet& pat1, const pat::Jet& pat2 ) {
    reco::CaloJet jet1( pat1.p4(), pat1.caloSpecific(), pat1.getJetConstituents() );
    reco::CaloJet jet2( pat2.p4(), pat2.caloSpecific(), pat2.getJetConstituents() );
    return ( jet1 == jet2 ); 
  }
}

// -----------------------------------------------------------------------------
//
namespace reco { 
  bool operator< ( const reco::GenJet& left, const reco::GenJet& right )  {
    return ( fabs ( left.pt() - right.pt() ) > std::numeric_limits<double>::epsilon() ?  
	     left.pt() > right.pt() :  //@@ NOTE THAT THIS IS EFFECTIVELY GREATER THAN!!!
	     false ); 
  }
}

#endif // JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h
