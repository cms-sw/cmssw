#include "JetMETCorrections/JetPlusTrack/plugins/RawPATJetProducer.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

// -----------------------------------------------------------------------------
//
RawPATJetProducer::RawPATJetProducer( const edm::ParameterSet& pset ) 
  : jets_( pset.getParameter<edm::InputTag>("JetCollection") )
{
  produces< std::vector<pat::Jet> >();
}

// -----------------------------------------------------------------------------
//
RawPATJetProducer::~RawPATJetProducer() {;}

// -----------------------------------------------------------------------------
//
void RawPATJetProducer::produce( edm::Event& event, const edm::EventSetup& setup ) {

  // Container for raw, uncorrected pat::Jets
  std::auto_ptr< std::vector<pat::Jet> > raw( new std::vector<pat::Jet> );
  raw->clear();

  // Retrieve pat::Jets
  edm::Handle< edm::View<pat::Jet> > jets;
  event.getByLabel( jets_, jets );

  // Match jets  
  edm::View<pat::Jet>::const_iterator ii = jets->begin();
  edm::View<pat::Jet>::const_iterator jj = jets->end();
  for ( ; ii != jj; ++ii ) {
    raw->push_back( ii->correctedJet("RAW") );
  }
  
  // Sort and put in Event
  NumericSafeGreaterByPt<pat::Jet> sorted;
  std::sort( raw->begin (), raw->end (), sorted );
  event.put(raw);

}

// -----------------------------------------------------------------------------
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RawPATJetProducer);
