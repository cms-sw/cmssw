#ifndef JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h
#define JetMETCorrections_JetPlusTracks_SortJetCollectionsByGenJetPt_h

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
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
  
    edm::InputTag matchedCaloJets_;
    edm::InputTag matchedGenJets_;
    edm::InputTag genJets_;
    std::vector<edm::InputTag> caloJets_;
    std::vector<edm::InputTag> patJets_;
  
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
  bool operator== ( const reco::CaloJet& jet1, const reco::CaloJet& jet2 ) {
    if ( true ) { // use det ids
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
    } else { // don't compare calo towers
      std::vector< edm::Ptr<CaloTower> > towers1 = jet1.getCaloConstituents();
      std::vector< edm::Ptr<CaloTower> > towers2 = jet2.getCaloConstituents();
      if ( towers1.empty() || 
	   towers2.empty() || 
	   towers1.size() != towers2.size() ) { return false; }
      std::vector< edm::Ptr<CaloTower> >::const_iterator ii = towers1.begin();
      std::vector< edm::Ptr<CaloTower> >::const_iterator jj = towers1.end();
      for ( ; ii != jj; ++ii ) {
	std::vector< edm::Ptr<CaloTower> >::const_iterator iii = towers2.begin();
	std::vector< edm::Ptr<CaloTower> >::const_iterator jjj = towers2.end();
	for ( ; iii != jjj; ++iii ) { if ( *iii == *ii ) { break; } }
	if ( iii == towers2.end() ) { return false; }
      }
      return true;
    }
  }
}

// -----------------------------------------------------------------------------
// just wraps the operator== method for reco::CaloJets
namespace pat {
  bool operator== ( const pat::Jet& pat1, const pat::Jet& pat2 ) {
    reco::CaloJet jet1( pat1.p4(), pat1.caloSpecific(), pat1.getJetConstituents() ); 
    reco::CaloJet jet2( pat2.p4(), pat2.caloSpecific(), pat2.getJetConstituents() ); 
/*     reco::CaloJet* calo1 = pat1.clone(); */
/*     reco::CaloJet* calo2 = pat2.clone(); */
/*     reco::CaloJet calo1( dynamic_cast<reco::CaloJet&>( const_cast<pat::Jet&>(pat1) ) ); */
/*     reco::CaloJet calo2( pat2 ); */
    return ( jet1 == jet2 ); 
/*     return ( calo1 == calo2 ); */
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
