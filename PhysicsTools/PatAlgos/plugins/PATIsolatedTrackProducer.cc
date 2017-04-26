#include <string>
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"

namespace pat {

    class PATIsolatedTrackProducer : public edm::EDProducer {
        public:
            explicit PATIsolatedTrackProducer(const edm::ParameterSet&);
            ~PATIsolatedTrackProducer();

            virtual void produce(edm::Event&, const edm::EventSetup&);

        private: 
            
            const edm::EDGetTokenT<pat::PackedCandidateCollection>    pc_;
            const float pT_cut;  // only save cands with pT>pT_cut
            const float dR_cut;  // isolation radius
            const float dZ_cut;  // save if either from PV or |dz|<dZ_cut
            const float absIso_cut;  // save if ANY of absIso, relIso, or miniRelIso pass the cuts 
            const float relIso_cut;
            const float miniRelIso_cut;
    };
}

pat::PATIsolatedTrackProducer::PATIsolatedTrackProducer(const edm::ParameterSet& iConfig) :
  pc_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedCandidates"))),
  pT_cut(iConfig.getParameter<double>("pT_cut")),
  dR_cut(iConfig.getParameter<double>("dR_cut")),
  dZ_cut(iConfig.getParameter<double>("dZ_cut")),
  absIso_cut(iConfig.getParameter<double>("absIso_cut")),
  relIso_cut(iConfig.getParameter<double>("relIso_cut")),
  miniRelIso_cut(iConfig.getParameter<double>("miniRelIso_cut"))
{
    produces< pat::IsolatedTrackCollection > ();
}

pat::PATIsolatedTrackProducer::~PATIsolatedTrackProducer() {}


void pat::PATIsolatedTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<pat::PackedCandidateCollection> pc_h;
    iEvent.getByToken( pc_, pc_h );
    const pat::PackedCandidateCollection *pc = pc_h.product();

    auto outPtrP = std::make_unique<std::vector<pat::IsolatedTrack>>();
    
    for(pat::PackedCandidateCollection::const_iterator pf_it = pc->begin(); pf_it != pc->end(); pf_it++){

        if(pf_it->p4().pt() < 5)
            continue;

        if(pf_it->fromPV()<=1 && fabs(pf_it->dz()) > dZ_cut)
            continue;

        if(pf_it->charge() == 0)
            continue;

        pat::PackedCandidatePtr pcref = refToPtr(pat::PackedCandidateRef(pc_h, (unsigned int)(pf_it - pc->begin())));

        // compute track isolation
        float trackIso = 0.0;
        MiniIsolation miniIso = {0,0,0,0};
        float miniDR = std::min(0.2, std::max(0.05, 10./pf_it->p4().pt()));
        for(pat::PackedCandidateCollection::const_iterator pf_it2 = pc->begin(); pf_it2 != pc->end(); pf_it2++){
            if(pf_it == pf_it2)
                continue;
            int id = abs(pf_it2->pdgId());
            bool fromPV = (pf_it2->fromPV()>1 || fabs(pf_it2->dz()) < dZ_cut);
            float dr = deltaR(pf_it->p4(), pf_it2->p4());

            // charged cands from PV get added to trackIso
            if(dr < dR_cut && id==211 && fromPV){
                trackIso += pf_it2->p4().pt();
            }
            // do the mini isolation
            if(dr < miniDR){
                if(id == 211 && fromPV)
                    miniIso.chiso += pf_it2->p4().pt();
                else if(id == 211)
                    miniIso.puiso += pf_it2->p4().pt();
                if(id == 130)
                    miniIso.nhiso += pf_it2->p4().pt();
                if(id == 22)
                    miniIso.phiso += pf_it2->p4().pt();
            }
        }

        if(trackIso < absIso_cut ||
           trackIso/pf_it->p4().pt() < relIso_cut ||
           (miniIso.chiso + miniIso.nhiso + miniIso.phiso)/pf_it->p4().pt() < miniRelIso_cut){
            outPtrP->push_back(pat::IsolatedTrack(trackIso, miniIso, pf_it->p4(), pf_it->pdgId(), pcref));
        }

    }


    iEvent.put(std::move(outPtrP));
}


using pat::PATIsolatedTrackProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATIsolatedTrackProducer);
