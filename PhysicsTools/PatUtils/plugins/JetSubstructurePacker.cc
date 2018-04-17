#include "PhysicsTools/PatUtils/interface/JetSubstructurePacker.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"

JetSubstructurePacker::JetSubstructurePacker(const edm::ParameterSet& iConfig) :
  distMax_( iConfig.getParameter<double>("distMax") ),
  jetToken_(consumes<edm::View<pat::Jet> >( iConfig.getParameter<edm::InputTag>("jetSrc") )),
  algoLabels_( iConfig.getParameter< std::vector<std::string> > ("algoLabels") ),
  algoTags_ (iConfig.getParameter<std::vector<edm::InputTag> > ( "algoTags" )),
  fixDaughters_(iConfig.getParameter<bool>("fixDaughters"))
{
  algoTokens_ =edm::vector_transform(algoTags_, [this](edm::InputTag const & tag){return consumes< edm::View<pat::Jet> >(tag);});
  if (fixDaughters_) {
    pf2pc_ = consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
    pc2pf_ = consumes<edm::Association<reco::PFCandidateCollection   > >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
  }
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

  auto outputs = std::make_unique<std::vector<pat::Jet>>();
 
  edm::Handle< edm::View<pat::Jet> > jetHandle;
  std::vector< edm::Handle< edm::View<pat::Jet> > > algoHandles;

  edm::Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc;
  edm::Handle<edm::Association<reco::PFCandidateCollection   > > pc2pf;
  if (fixDaughters_) {
    iEvent.getByToken(pf2pc_,pf2pc);
    iEvent.getByToken(pc2pf_,pc2pf);
  }

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
      float dRMin = distMax_;
      for ( auto const & jjet : *ialgoHandle ) {       
	if ( reco::deltaR( ijet, jjet ) < dRMin ) {
	  for ( auto const & userfloatstr : jjet.userFloatNames() ) {
	    outputs->back().addUserFloat( userfloatstr, jjet.userFloat(userfloatstr) );
	  }
	  for ( auto const & userintstr : jjet.userIntNames() ) {
	    outputs->back().addUserInt( userintstr, jjet.userInt(userintstr) );
	  }
	  for ( auto const & usercandstr : jjet.userCandNames() ) {
	    outputs->back().addUserCand( usercandstr, jjet.userCand(usercandstr) );
	  }
	  for ( size_t ida = 0; ida < jjet.numberOfDaughters(); ++ida ) {
	    reco::CandidatePtr candPtr =  jjet.daughterPtr( ida);
	    nextSubjets.push_back( edm::Ptr<pat::Jet> ( candPtr ) );
	  }
	  break;
	}
      }
      outputs->back().addSubjets( nextSubjets, algoLabels_[index] );
      ++index; 
    }

    // fix daughters
    if (fixDaughters_) {

        std::vector<reco::CandidatePtr> daughtersInSubjets;
        std::vector<reco::CandidatePtr> daughtersNew;
        const std::vector<reco::CandidatePtr> & jdausPF = outputs->back().daughterPtrVector();
	std::vector<reco::CandidatePtr> jdaus;
	jdaus.reserve( jdausPF.size() );
	// Convert the daughters to packed candidates. This is easier than ref-navigating through PUPPI or CHS to particleFlow.
	for ( auto const & jdau : jdausPF ) {
	  jdaus.push_back( edm::refToPtr((*pf2pc)[jdau]) );
	}
		
        for ( const edm::Ptr<pat::Jet> & subjet : outputs->back().subjets()) {
            const std::vector<reco::CandidatePtr> & sjdaus = subjet->daughterPtrVector();
            // check that the subjet does not contain any extra constituents not contained in the jet
            bool skipSubjet = false;
            for (const reco::CandidatePtr & dau : sjdaus) {
                if (std::find(jdaus.begin(), jdaus.end(), dau) == jdaus.end()) {
                    skipSubjet = true;
                    break;
                }
            }
            if (skipSubjet) continue;

            daughtersInSubjets.insert(daughtersInSubjets.end(), sjdaus.begin(), sjdaus.end());
            daughtersNew.push_back( reco::CandidatePtr(subjet) );
        }
        for (const reco::CandidatePtr & dau : jdaus) {
            if (std::find(daughtersInSubjets.begin(), daughtersInSubjets.end(), dau) == daughtersInSubjets.end()) {
                daughtersNew.push_back( dau );
            }
        }
        outputs->back().clearDaughters();
        for (const auto & dau : daughtersNew) outputs->back().addDaughter(dau);
    }
  }

  iEvent.put(std::move(outputs));

}

//define this as a plug-in
DEFINE_FWK_MODULE(JetSubstructurePacker);
