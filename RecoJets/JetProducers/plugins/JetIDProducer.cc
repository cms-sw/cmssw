#include "RecoJets/JetProducers/plugins/JetIDProducer.h"
#include "DataFormats/JetReco/interface/JetID.h"

#include <vector>

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
JetIDProducer::JetIDProducer(const edm::ParameterSet& iConfig) :
  src_       ( iConfig.getParameter<edm::InputTag>("src") ),
  helper_    ( iConfig, consumesCollector() ),
  muHelper_  ( iConfig, consumesCollector() )
{
  produces< reco::JetIDValueMap >();

  input_jet_token_ = consumes<edm::View<reco::CaloJet> >(src_);

}


JetIDProducer::~JetIDProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get the input jets
  edm::Handle< edm::View<reco::CaloJet> > h_jets;
  iEvent.getByToken( input_jet_token_, h_jets );

  // allocate the jet--->jetid value map
  std::auto_ptr<reco::JetIDValueMap> jetIdValueMap( new reco::JetIDValueMap );
  // instantiate the filler with the map
  reco::JetIDValueMap::Filler filler(*jetIdValueMap);
  
  // allocate the vector of ids
  size_t njets = h_jets->size();
  std::vector<reco::JetID>  ids (njets);
   
  // loop over the jets
  for ( edm::View<reco::CaloJet>::const_iterator jetsBegin = h_jets->begin(),
	  jetsEnd = h_jets->end(),
	  ijet = jetsBegin;
	ijet != jetsEnd; ++ijet ) {

    // get the id from each jet
    helper_.calculate( iEvent, *ijet );

    muHelper_.calculate( iEvent, iSetup, *ijet );

    ids[ijet-jetsBegin].fHPD               =  helper_.fHPD();
    ids[ijet-jetsBegin].fRBX               =  helper_.fRBX();
    ids[ijet-jetsBegin].n90Hits            =  helper_.n90Hits();
    ids[ijet-jetsBegin].fSubDetector1      =  helper_.fSubDetector1();
    ids[ijet-jetsBegin].fSubDetector2      =  helper_.fSubDetector2();
    ids[ijet-jetsBegin].fSubDetector3      =  helper_.fSubDetector3();
    ids[ijet-jetsBegin].fSubDetector4      =  helper_.fSubDetector4();
    ids[ijet-jetsBegin].restrictedEMF      =  helper_.restrictedEMF();
    ids[ijet-jetsBegin].nHCALTowers        =  helper_.nHCALTowers();
    ids[ijet-jetsBegin].nECALTowers        =  helper_.nECALTowers();
    ids[ijet-jetsBegin].approximatefHPD    =  helper_.approximatefHPD();
    ids[ijet-jetsBegin].approximatefRBX    =  helper_.approximatefRBX();
    ids[ijet-jetsBegin].hitsInN90          =  helper_.hitsInN90();    

    ids[ijet-jetsBegin].numberOfHits2RPC   = muHelper_.numberOfHits2RPC();
    ids[ijet-jetsBegin].numberOfHits3RPC   = muHelper_.numberOfHits3RPC();
    ids[ijet-jetsBegin].numberOfHitsRPC    = muHelper_.numberOfHitsRPC();
    
    ids[ijet-jetsBegin].fEB     = helper_.fEB   ();
    ids[ijet-jetsBegin].fEE     = helper_.fEE   ();
    ids[ijet-jetsBegin].fHB     = helper_.fHB   (); 
    ids[ijet-jetsBegin].fHE     = helper_.fHE   (); 
    ids[ijet-jetsBegin].fHO     = helper_.fHO   (); 
    ids[ijet-jetsBegin].fLong   = helper_.fLong ();
    ids[ijet-jetsBegin].fShort  = helper_.fShort();
    ids[ijet-jetsBegin].fLS     = helper_.fLSbad   ();
    ids[ijet-jetsBegin].fHFOOT  = helper_.fHFOOT();
  }
  
  // set up the map
  filler.insert( h_jets, ids.begin(), ids.end() );
 
  // fill the vals
  filler.fill();

  // write map to the event
  iEvent.put( jetIdValueMap );
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetIDProducer);
