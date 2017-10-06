//
// $Id: PATGenJetSlimmer.cc,v 1.1 2011/03/24 18:45:45 mwlebour Exp $
//

/**
  \class    pat::PATGenJetSlimmer PATGenJetSlimmer.h "PhysicsTools/PatAlgos/interface/PATGenJetSlimmer.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
*/

#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"

namespace pat {

  class PATGenJetSlimmer : public edm::stream::EDProducer<> {
  public:
    explicit PATGenJetSlimmer(const edm::ParameterSet & iConfig);
    ~PATGenJetSlimmer() override { }
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;
    
  private:
    const edm::EDGetTokenT<edm::View<reco::GenJet> > src_;
    const edm::EDGetTokenT<edm::Association<std::vector<pat::PackedGenParticle> > > gp2pgp_;
    
    const StringCutObjectSelector<reco::GenJet> cut_;
    const StringCutObjectSelector<reco::GenJet> cutLoose_;
    const unsigned nLoose_;
    
    /// reset daughters to an empty vector
    const bool clearDaughters_;
    /// drop the specific
    const bool dropSpecific_;
  };

} // namespace


pat::PATGenJetSlimmer::PATGenJetSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src"))),
    gp2pgp_(consumes<edm::Association<std::vector<pat::PackedGenParticle> > >(iConfig.getParameter<edm::InputTag>("packedGenParticles"))),
    cut_(iConfig.getParameter<std::string>("cut")),
    cutLoose_(iConfig.getParameter<std::string>("cutLoose")),
    nLoose_(iConfig.getParameter<unsigned>("nLoose")),
    clearDaughters_(iConfig.getParameter<bool>("clearDaughters")),
    dropSpecific_(iConfig.getParameter<bool>("dropSpecific"))
{
    produces<std::vector<reco::GenJet> >();
    produces< edm::Association<std::vector<reco::GenJet> > >("slimmedGenJetAssociation");
}

void 
pat::PATGenJetSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<reco::GenJet> >      src;
    iEvent.getByToken(src_, src);

    auto out = std::make_unique<vector<reco::GenJet> >();
    out->reserve(src->size());

    Handle<edm::Association<std::vector<pat::PackedGenParticle> > > gp2pgp;
    iEvent.getByToken(gp2pgp_,gp2pgp);

    auto genJetSlimmedGenJetAssociation = make_unique< edm::Association<std::vector<reco::GenJet> > > ();

    auto mapping = std::make_unique<std::vector<int> >();
    mapping->reserve(src->size());

    unsigned nl = 0; // number of loose jets
    for (View<reco::GenJet>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {

	bool selectedLoose = false;
	if ( nLoose_ > 0 && nl < nLoose_ && cutLoose_(*it) ) {
	  selectedLoose = true;
	  ++nl;
	}

	bool pass = cut_(*it) || selectedLoose;
        if (!pass ) {
            mapping->push_back(-1);
            continue;
        }

        out->push_back(*it);
        reco::GenJet & jet = out->back();

        mapping->push_back(it-src->begin());


        if (clearDaughters_) {
            jet.clearDaughters();
        }   
	    else // rekey   
	    {
                //copy old 
		reco::CompositePtrCandidate::daughters old = jet.daughterPtrVector();
		jet.clearDaughters();
		std::map<unsigned int,reco::CandidatePtr> ptrs;
		for(unsigned int  i=0;i<old.size();i++)
		{
//	if(! ((*gp2pgp)[old[i]]).isNonnull())	{
//		std::cout << "Missing ref for key"  <<  old[i].key() << " pdgid " << old[i]->pdgId() << " st "<<   old[i]->status() <<  " pt " << old[i]->pt() << " eta " << old[i]->eta() << std::endl;
//	}
			ptrs[((*gp2pgp)[old[i]]).key()]=refToPtr((*gp2pgp)[old[i]]);
		}
		for(std::map<unsigned int,reco::CandidatePtr>::iterator itp=ptrs.begin();itp!=ptrs.end();itp++) //iterate on sorted items
		{
			jet.addDaughter(itp->second);
		}


	}
        if (dropSpecific_) {
            jet.setSpecific( reco::GenJet::Specific() );
        }
        
    }

    edm::OrphanHandle<std::vector<reco::GenJet> >  orphanHandle= iEvent.put(std::move(out));

    auto asso = std::make_unique<edm::Association<std::vector<reco::GenJet> > >(orphanHandle);
    edm::Association< std::vector<reco::GenJet> >::Filler slimmedAssoFiller(*asso);
    slimmedAssoFiller.insert(src, mapping->begin(), mapping->end());
    slimmedAssoFiller.fill();

    
    iEvent.put(std::move(asso),"slimmedGenJetAssociation");
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATGenJetSlimmer);
