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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

namespace pat {

  class PATJetSlimmer : public edm::stream::EDProducer<> {
    public:
      explicit PATJetSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATJetSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup&) override final;

    private:
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      edm::EDGetTokenT<edm::ValueMap<reco::CandidatePtr>> pf2pcAny_;
      const edm::EDGetTokenT<edm::View<pat::Jet> >  jets_;
      const StringCutObjectSelector<pat::Jet> dropJetVars_,dropDaughters_,rekeyDaughters_,dropTrackRefs_,dropSpecific_,dropTagInfos_;
      const bool modifyJet_, mayNeedDaughterMap_, mixedDaughters_;
      std::unique_ptr<pat::ObjectModifier<pat::Jet> > jetModifier_;
  };

} // namespace


pat::PATJetSlimmer::PATJetSlimmer(const edm::ParameterSet & iConfig) :
    jets_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("src"))),
    dropJetVars_(iConfig.getParameter<std::string>("dropJetVars")),
    dropDaughters_(iConfig.getParameter<std::string>("dropDaughters")),
    rekeyDaughters_(iConfig.getParameter<std::string>("rekeyDaughters")),
    dropTrackRefs_(iConfig.getParameter<std::string>("dropTrackRefs")),
    dropSpecific_(iConfig.getParameter<std::string>("dropSpecific")),
    dropTagInfos_(iConfig.getParameter<std::string>("dropTagInfos")),
    modifyJet_(iConfig.getParameter<bool>("modifyJets")),
    mayNeedDaughterMap_(iConfig.getParameter<std::string>("dropDaughters") != "1" && iConfig.getParameter<std::string>("rekeyDaughters") != "0"),
    mixedDaughters_(iConfig.getParameter<bool>("mixedDaughters"))
{
    if (mayNeedDaughterMap_) {
        if (mixedDaughters_) {
            pf2pcAny_ = consumes<edm::ValueMap<reco::CandidatePtr> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
        } else {
            pf2pc_ = consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
        }
    }
    edm::ConsumesCollector sumes(consumesCollector());
    if( modifyJet_ ) {
      const edm::ParameterSet& mod_config = iConfig.getParameter<edm::ParameterSet>("modifierConfig");
      jetModifier_.reset(new pat::ObjectModifier<pat::Jet>(mod_config) );
      jetModifier_->setConsumes(sumes);
    } else {
      jetModifier_.reset(nullptr);
    }
    produces<std::vector<pat::Jet> >();
}

void 
pat::PATJetSlimmer::beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup& iSetup) {
  if( modifyJet_ ) jetModifier_->setEventContent(iSetup);
}

void 
pat::PATJetSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Jet> >      src;
    iEvent.getByToken(jets_, src);
    Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc;
    Handle<edm::ValueMap<reco::CandidatePtr> > pf2pcAny;
    if (mayNeedDaughterMap_) {
        if (mixedDaughters_) {
            iEvent.getByToken(pf2pcAny_,pf2pcAny);
        } else {
            iEvent.getByToken(pf2pc_,pf2pc);
        }
    }
	
    auto out = std::make_unique<std::vector<pat::Jet>>();
    out->reserve(src->size());

    if( modifyJet_ ) { jetModifier_->setEvent(iEvent); }

    for (edm::View<pat::Jet>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
	    out->push_back(*it);
	    pat::Jet & jet = out->back();

            if( modifyJet_ ) { jetModifier_->modify(jet); }
            
	    if(dropTagInfos_(*it)){
		    jet.tagInfoLabels_.clear();
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
	    } else if (rekeyDaughters_(*it)) {  //rekey
		    //copy old 
		    reco::CompositePtrCandidate::daughters old = jet.daughterPtrVector();
		    jet.clearDaughters();
                    if (mixedDaughters_) {
                        std::vector<reco::CandidatePtr> ptrs;	    
                        for(const reco::CandidatePtr &oldptr : old) {
                            ptrs.push_back( (*pf2pcAny)[oldptr] );
                        }
                        std::sort(ptrs.begin(), ptrs.end());
                        for(const reco::CandidatePtr &newptr : ptrs) {
                            jet.addDaughter(newptr);
                        }
                    } else {
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

    iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATJetSlimmer);
