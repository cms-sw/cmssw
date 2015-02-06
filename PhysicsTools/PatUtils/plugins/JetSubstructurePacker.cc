#include "PhysicsTools/PatUtils/interface/JetSubstructurePacker.h"
#include "DataFormats/Math/interface/deltaR.h"

JetSubstructurePacker::JetSubstructurePacker(const edm::ParameterSet& iConfig) :
  distMin_( iConfig.getParameter<double>("distMin") ),
  jetToken_(consumes<edm::View<pat::Jet> >( iConfig.getParameter<edm::InputTag>("jetSrc") )),
  algoLabels_( iConfig.getParameter< std::vector<std::string> > ("algoLabels") ),
  algoTags_ (iConfig.getParameter<std::vector<edm::InputTag> > ( "algoTags" ))
{
  algoTokens_ =edm::vector_transform(algoTags_, [this](edm::InputTag const & tag){return consumes< edm::View<pat::Jet> >(tag);});
  //register products
  produces<std::vector<pat::Jet> > ();
}


JetSubstructurePacker::~JetSubstructurePacker()
{
}


// ------------ method called to produce the data  ------------
void
JetSubstructurePacker::produce(edm::Event& iEvent, const edm::EventSetup&)
{  

  std::auto_ptr< std::vector<pat::Jet> > outputs( new std::vector<pat::Jet> );
 
  edm::Handle< edm::View<pat::Jet> > jetHandle;
  std::vector< edm::Handle< edm::View<pat::Jet> > > algoHandles;

  iEvent.getByToken( jetToken_, jetHandle );
  algoHandles.resize( algoTags_.size() );
  for ( size_t i = 0; i < algoTags_.size(); ++i ) {
    iEvent.getByToken( algoTokens_[i], algoHandles[i] ); 
  }

  // Loop over the input jets that will be modified.
  for ( auto const & ijet : *jetHandle  ) {
    // Copy the jet.
    outputs->push_back( ijet );

    // Loop over the substructure collections
    unsigned int index = 0;
    for ( auto const & ialgoHandle : algoHandles ) {      
      std::vector< edm::Ptr<pat::Jet> > nextSubjets;
      float dRMin = distMin_;

      for ( auto const & jjet : *ialgoHandle ) {
	
	if ( reco::deltaR( ijet, jjet ) < dRMin ) {
	  for ( size_t ida = 0; ida < jjet.numberOfDaughters(); ++ida ) {

	    edm::Ptr<reco::Candidate> candPtr =  jjet.daughterPtr( ida);
	    nextSubjets.push_back( edm::Ptr<pat::Jet> ( candPtr ) );
	  }
	}
	break;
      }

      outputs->back().addSubjets( nextSubjets, algoLabels_[index] );
      ++index; 
    }
  }

  iEvent.put(outputs);

}

//define this as a plug-in
DEFINE_FWK_MODULE(JetSubstructurePacker);
