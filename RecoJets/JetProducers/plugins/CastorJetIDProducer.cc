#include "RecoJets/JetProducers/plugins/CastorJetIDProducer.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"

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
CastorJetIDProducer::CastorJetIDProducer(const edm::ParameterSet& iConfig) :
  src_       ( iConfig.getParameter<edm::InputTag>("src") ),
  helper_    ( )
{
  produces< reco::CastorJetIDValueMap >();

  input_jet_token_ = consumes<edm::View<reco::BasicJet> >(src_);

}


CastorJetIDProducer::~CastorJetIDProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CastorJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get the input jets
  edm::Handle< edm::View<reco::BasicJet> > h_jets;
  iEvent.getByToken( input_jet_token_, h_jets );

  // allocate the jet--->jetid value map
  std::auto_ptr<reco::CastorJetIDValueMap> castorjetIdValueMap( new reco::CastorJetIDValueMap );
  // instantiate the filler with the map
  reco::CastorJetIDValueMap::Filler filler(*castorjetIdValueMap);
  
  // allocate the vector of ids
  size_t njets = h_jets->size();
  std::vector<reco::CastorJetID>  ids (njets);
   
  // loop over the jets
  for ( edm::View<reco::BasicJet>::const_iterator jetsBegin = h_jets->begin(),
	  jetsEnd = h_jets->end(),
	  ijet = jetsBegin;
	ijet != jetsEnd; ++ijet ) {

    // get the id from each jet
    helper_.calculate( iEvent, *ijet );

    ids[ijet-jetsBegin].emEnergy               =  helper_.emEnergy();
    ids[ijet-jetsBegin].hadEnergy               =  helper_.hadEnergy();
    ids[ijet-jetsBegin].fem            =  helper_.fem();
    ids[ijet-jetsBegin].depth      =  helper_.depth();
    ids[ijet-jetsBegin].width      =  helper_.width();
    ids[ijet-jetsBegin].fhot      =  helper_.fhot();
    ids[ijet-jetsBegin].sigmaz      =  helper_.sigmaz();
    ids[ijet-jetsBegin].nTowers      =  helper_.nTowers(); 


  }
  
  // set up the map
  filler.insert( h_jets, ids.begin(), ids.end() );
 
  // fill the vals
  filler.fill();

  // write map to the event
  iEvent.put( castorjetIdValueMap );
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorJetIDProducer);
