//
// $Id: PATJetSlimmer.cc,v 1.1 2011/03/24 18:45:45 mwlebour Exp $
//

/**
  \class    pat::PATJetSlimmer PATJetSlimmer.h "PhysicsTools/PatAlgos/interface/PATJetSlimmer.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
*/

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace pat {

  class PATJetSlimmer : public edm::EDProducer {
    public:
      explicit PATJetSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATJetSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::View<pat::Jet> > src_;
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>  > map_;
      
      /// clear mJetArea, mPassNumber, mPileupEnergy
      bool clearJetVars_;
      /// reset daughters to an empty vector
      bool clearDaughters_;
      bool clearTrackRefs_;
//       /// reduce GenJet to a bare 4-vector
//       bool slimGenJet_;
      /// drop the Calo or PF specific
      bool dropSpecific_;
//       /// drop the JetCorrFactors (but keep the jet corrected!)
//       bool dropJetCorrFactors_;
  };

} // namespace

pat::PATJetSlimmer::PATJetSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("src"))),
    map_(consumes<edm::Association<pat::PackedCandidateCollection>  >(iConfig.getParameter<edm::InputTag>("map"))),
    clearJetVars_(iConfig.getParameter<bool>("clearJetVars")),
    clearDaughters_(iConfig.getParameter<bool>("clearDaughters")),
    clearTrackRefs_(iConfig.getParameter<bool>("clearTrackRefs")),
    dropSpecific_(iConfig.getParameter<bool>("dropSpecific"))
//     dropJetCorrFactors_(iConfig.getParameter<bool>("dropJetCorrFactors"))
{
    produces<std::vector<pat::Jet> >();
}

void 
pat::PATJetSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Jet> >      src;
    iEvent.getByToken(src_, src);
    Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc;
    iEvent.getByToken(map_,pf2pc);
	
    auto_ptr<vector<pat::Jet> >  out(new vector<pat::Jet>());
    out->reserve(src->size());

    for (edm::View<pat::Jet>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Jet & jet = out->back();

        if (clearJetVars_) {
//             jet.setJetArea(0); 
            jet.setNPasses(0);
//             jet.setPileup(0);
        }
	if(clearTrackRefs_)
	{
	   jet.setAssociatedTracks(reco::TrackRefVector());		
	}
        if (clearDaughters_) {
            jet.clearDaughters();
            jet.pfCandidatesFwdPtr_.clear();
            jet.caloTowersFwdPtr_.clear();
        } else {  //rekey
	    //copy old 
	    reco::CompositePtrCandidate::daughters old = jet.daughterPtrVector();
            jet.clearDaughters();
	    std::map<unsigned int,reco::CandidatePtr> ptrs;	    
	    for(unsigned int  i=0;i<old.size();i++)
	    {
	//	jet.addDaughter(refToPtr((*pf2pc)[old[i]]));
		ptrs[((*pf2pc)[old[i]]).key()]=refToPtr((*pf2pc)[old[i]]);
	    }
	    for(std::map<unsigned int,reco::CandidatePtr>::iterator itp=ptrs.begin();itp!=ptrs.end();itp++) //iterate on sorted items
	    {
		jet.addDaughter(itp->second);
	    }
		

	}	
//         if (slimGenJet_) {
//             const reco::GenJet * genjet = it->genJet();
//             if (genjet) {
//                 std::vector<reco::GenJet> tempGenJet(1, reco::GenJet(genjet->p4(), reco::Particle::Point(), reco::GenJet::Specific()));
//                 jet.setGenJet(reco::GenJetRef(&tempGenJet,0), true);
//             }
//         }
        if (dropSpecific_) {
            // FIXME add method in pat::Jet
            jet.specificCalo_.clear();    
            jet.specificPF_.clear();    
        }
//         if (dropJetCorrFactors_) {
//             // FIXME add method in pat::Jet
//             jet.jetEnergyCorrections_.clear();
//         }
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATJetSlimmer);
