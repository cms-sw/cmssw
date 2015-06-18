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

  std::auto_ptr< std::vector<pat::Jet> > outputs( new std::vector<pat::Jet> );
 
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
        const std::vector<reco::CandidatePtr> & jdaus = outputs->back().daughterPtrVector();
        //std::cout << "Jet with pt " << outputs->back().pt() << ", " << outputs->back().numberOfDaughters() << " daughters, " << outputs->back().subjets().size() << ", subjets" << std::endl;
        for ( const edm::Ptr<pat::Jet> & subjet : outputs->back().subjets()) {
            const std::vector<reco::CandidatePtr> & sjdaus = subjet->daughterPtrVector();

            // check that the subjet does not contain any extra constituents not contained in the jet
            bool skipSubjet = false;
            for (const reco::CandidatePtr & dau : sjdaus) {
                reco::CandidatePtr rekeyed = edm::refToPtr((*pc2pf)[dau]);
                if (std::find(jdaus.begin(), jdaus.end(), rekeyed) == jdaus.end()) {
                    skipSubjet = true;
                    break;
                }
            }
            if (skipSubjet) continue;

            daughtersInSubjets.insert(daughtersInSubjets.end(), sjdaus.begin(), sjdaus.end());
            daughtersNew.push_back( reco::CandidatePtr(subjet) );
            //std::cout << "     found  " << subjet->numberOfDaughters() << " daughters in a subjet" << std::endl;
        }
        //if (!daughtersInSubjets.empty()) std::cout << "     subjet daughters are from collection " << daughtersInSubjets.front().id() << std::endl;
        //std::cout << "     in total,  " << daughtersInSubjets.size() << " daughters from subjets" << std::endl;
        for (const reco::CandidatePtr & dau : jdaus) {
            //if (!pf2pc->contains(dau.id())) {
            //    std::cout << "     daughter from collection " << dau.id() << " not in the value map!" << std::endl;
            //    std::cout << "     map expects collection " << pf2pc->ids().front().first << std::endl;
            //    continue;
            //}
            reco::CandidatePtr rekeyed = edm::refToPtr((*pf2pc)[dau]);
            if (std::find(daughtersInSubjets.begin(), daughtersInSubjets.end(), rekeyed) == daughtersInSubjets.end()) {
                daughtersNew.push_back( rekeyed );
            }
        }
        //std::cout << "     in total,  " << daughtersNew.size() << " daughters including subjets" << std::endl;
        //if (daughtersNew.size() + daughtersInSubjets.size() - outputs->back().subjets().size() == outputs->back().numberOfDaughters()) {
        //    std::cout << "     it all adds up to the original number of daughters" << std::endl;
        //}
        outputs->back().clearDaughters();
        for (const auto & dau : daughtersNew) outputs->back().addDaughter(dau);
    }
  }

  iEvent.put(outputs);

}

//define this as a plug-in
DEFINE_FWK_MODULE(JetSubstructurePacker);
