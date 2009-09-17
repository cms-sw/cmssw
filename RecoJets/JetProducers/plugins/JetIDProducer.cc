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
  helper_    ( iConfig )
{
  produces< reco::JetIDValueMap >();
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
  iEvent.getByLabel( src_, h_jets );

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
  }
  
  // set up the map
  filler.insert( h_jets, ids.begin(), ids.end() );
 
  // fill the vals
  filler.fill();

  // write map to the event
  iEvent.put( jetIdValueMap );
}

// ------------ method called once each job just before starting event loop  ------------
void 
JetIDProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetIDProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetIDProducer);
