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
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace pat {

  class PATJetSlimmer : public edm::EDProducer {
    public:
      explicit PATJetSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATJetSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      edm::EDGetTokenT<edm::View<pat::Jet> >  jets_;
      StringCutObjectSelector<pat::Jet> dropJetVars_,dropDaughters_,dropTrackRefs_,dropSpecific_,dropTagInfos_;
  };

} // namespace


pat::PATJetSlimmer::PATJetSlimmer(const edm::ParameterSet & iConfig) :
    pf2pc_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
    jets_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("src"))),
    dropJetVars_(iConfig.getParameter<std::string>("dropJetVars")),
    dropDaughters_(iConfig.getParameter<std::string>("dropDaughters")),
    dropTrackRefs_(iConfig.getParameter<std::string>("dropTrackRefs")),
    dropSpecific_(iConfig.getParameter<std::string>("dropSpecific")),
    dropTagInfos_(iConfig.getParameter<std::string>("dropTagInfos"))
{
    produces<std::vector<pat::Jet> >();
}

void 
pat::PATJetSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Jet> >      src;
    iEvent.getByToken(jets_, src);
    Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc;
    iEvent.getByToken(pf2pc_,pf2pc);
	
    auto_ptr<vector<pat::Jet> >  out(new vector<pat::Jet>());
    out->reserve(src->size());

    for (edm::View<pat::Jet>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
	    out->push_back(*it);
	    pat::Jet & jet = out->back();
	    if(dropTagInfos_(*it)){
		    jet.tagInfos_.clear();
		    jet.tagInfosFwdPtr_.clear(); 
	    }
	    if (dropJetVars_(*it)) {
		    //             jet.setJetArea(0); 
		    jet.setNPasses(0);
		    //             jet.setPileup(0);
	    }
	    if(dropTrackRefs_(*it))
	    {
		    jet.setAssociatedTracks(reco::TrackRefVector());		
	    }
	    if (dropDaughters_(*it)) {
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
	    if (dropSpecific_(*it)) {
		    // FIXME add method in pat::Jet
		    jet.specificCalo_.clear();    
		    jet.specificPF_.clear();    
	    }
	    //         if (dropJetCorrFactors_(*it)) {
	    //             // FIXME add method in pat::Jet
	    //             jet.jetEnergyCorrections_.clear();
	    //         }
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATJetSlimmer);
