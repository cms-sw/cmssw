#include "PhysicsTools/PatUtils/interface/BoostedJetMerger.h"


BoostedJetMerger::BoostedJetMerger(const edm::ParameterSet& iConfig) :
  jetToken_(consumes<edm::View<pat::Jet> >( iConfig.getParameter<edm::InputTag>("jetSrc") )),
  subjetToken_(consumes<edm::View<pat::Jet> >( iConfig.getParameter<edm::InputTag>("subjetSrc") ))
{
  //register products
  produces<std::vector<pat::Jet> > ();
  produces<std::vector<pat::Jet> > ("SubJets");
}


BoostedJetMerger::~BoostedJetMerger()
{
}


// ------------ method called to produce the data  ------------
void
BoostedJetMerger::produce(edm::Event& iEvent, const edm::EventSetup&)
{  

  std::auto_ptr< std::vector<pat::Jet> > outputs( new std::vector<pat::Jet> );
  std::auto_ptr< std::vector<pat::Jet> > outputSubjets( new std::vector<pat::Jet> );

  edm::RefProd< std::vector<pat::Jet> > h_subJetsOut = iEvent.getRefBeforePut< std::vector<pat::Jet> >( "SubJets" );
 
  edm::Handle< edm::View<pat::Jet> > jetHandle;
  edm::Handle< edm::View<pat::Jet> > subjetHandle;

  iEvent.getByToken( jetToken_, jetHandle );
  iEvent.getByToken( subjetToken_, subjetHandle ); 

  for ( edm::View<pat::Jet>::const_iterator ijetBegin = jetHandle->begin(),
	  ijetEnd = jetHandle->end(), ijet = ijetBegin; ijet != ijetEnd; ++ijet ) {
    
    outputs->push_back( *ijet );
    std::vector< edm::Ptr<reco::Candidate> > nextSubjets;

    for ( unsigned int isubjet = 0; isubjet < ijet->numberOfDaughters(); ++isubjet ) {
      edm::Ptr<reco::Candidate> const & subjet = ijet->daughterPtr(isubjet);
      edm::View<pat::Jet>::const_iterator ifound = find_if( subjetHandle->begin(),
							    subjetHandle->end(),
							    FindCorrectedSubjet(subjet) );
      if ( ifound != subjetHandle->end() ) {

	outputSubjets->push_back( *ifound );

	edm::Ref<std::vector<pat::Jet> > subjetRef ( h_subJetsOut, outputSubjets->size() - 1);
	edm::Ptr< pat::Jet > subjetPtr ( h_subJetsOut.id(), subjetRef.key(), h_subJetsOut.productGetter() );
	nextSubjets.push_back( subjetPtr );
      }
    }
    outputs->back().clearDaughters();
    for ( std::vector< edm::Ptr<reco::Candidate> >::const_iterator nextSubjet = nextSubjets.begin(),
	    nextSubjetEnd = nextSubjets.end(); nextSubjet != nextSubjetEnd; ++nextSubjet ) {
      outputs->back().addDaughter( *nextSubjet );
    }

    
  }

  
  iEvent.put(outputs);
  iEvent.put(outputSubjets, "SubJets");

}

//define this as a plug-in
DEFINE_FWK_MODULE(BoostedJetMerger);
