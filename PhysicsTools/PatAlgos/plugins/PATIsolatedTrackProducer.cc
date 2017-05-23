#include <string>
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace pat {

    class PATIsolatedTrackProducer : public edm::global::EDProducer<> {
        public:
            explicit PATIsolatedTrackProducer(const edm::ParameterSet&);
            ~PATIsolatedTrackProducer();

        virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

        private: 
            
            const edm::EDGetTokenT<pat::PackedCandidateCollection>    pc_;
            const float pT_cut;  // only save cands with pT>pT_cut
            const float dR_cut;  // isolation radius
            const float dZ_cut;  // save if either from PV or |dz|<dZ_cut
            const std::vector<double> miniIsoParams;
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
  miniIsoParams(iConfig.getParameter<std::vector<double> >("miniIsoParams")),
  absIso_cut(iConfig.getParameter<double>("absIso_cut")),
  relIso_cut(iConfig.getParameter<double>("relIso_cut")),
  miniRelIso_cut(iConfig.getParameter<double>("miniRelIso_cut"))
{
    produces< pat::IsolatedTrackCollection > ();
}

pat::PATIsolatedTrackProducer::~PATIsolatedTrackProducer() {}


void pat::PATIsolatedTrackProducer::produce(edm::StreamID stream, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

    edm::Handle<pat::PackedCandidateCollection> pc_h;
    iEvent.getByToken( pc_, pc_h );
    const pat::PackedCandidateCollection *pc = pc_h.product();

    auto outPtrP = std::make_unique<std::vector<pat::IsolatedTrack>>();
    
    for(pat::PackedCandidateCollection::const_iterator pf_it = pc->begin(); pf_it != pc->end(); pf_it++){

        if(pf_it->p4().pt() < pT_cut)
            continue;

        if(pf_it->fromPV()<=1 && fabs(pf_it->dz()) > dZ_cut)
            continue;

        if(pf_it->charge() == 0)
            continue;

        pat::PackedCandidateRef pcref = pat::PackedCandidateRef(pc_h, (unsigned int)(pf_it - pc->begin()));

        // compute both standard and mini PFIsolation
        // mini-isolation reference: https://hypernews.cern.ch/HyperNews/CMS/get/susy/1991.html
        float chiso=0, nhiso=0, phiso=0, puiso=0;   // standard isolation
        float chmiso=0, nhmiso=0, phmiso=0, pumiso=0;  // mini isolation
        float miniDR = std::max(miniIsoParams[0], std::min(miniIsoParams[1], miniIsoParams[2]/pf_it->p4().pt()));
        for(pat::PackedCandidateCollection::const_iterator pf_it2 = pc->begin(); pf_it2 != pc->end(); pf_it2++){
            if(pf_it == pf_it2)
                continue;
            int id = std::abs(pf_it2->pdgId());
            bool fromPV = (pf_it2->fromPV()>1 || fabs(pf_it2->dz()) < dZ_cut);
            float pt = pf_it2->p4().pt();
            float dr = deltaR(pf_it->p4(), pf_it2->p4());

            if(dr < dR_cut){
                // charged cands from PV get added to trackIso
                if(id==211 && fromPV)
                    chiso += pt;
                // charged cands not from PV get added to pileup iso
                else if(id==211)
                    puiso += pt;
                // neutral hadron iso
                if(id==130)
                    nhiso += pt;
                // photon iso
                if(id==22)
                    phiso += pt;
            }
            // same for mini isolation
            if(dr < miniDR){
                if(id == 211 && fromPV)
                    chmiso += pt;
                else if(id == 211)
                    pumiso += pt;
                if(id == 130)
                    nhmiso += pt;
                if(id == 22)
                    phmiso += pt;
            }
        }
        
        pat::PFIsolation isolationDR03(chiso, nhiso, phiso, puiso);
        pat::PFIsolation miniIso(chmiso, nhmiso, phmiso, pumiso);

        if(isolationDR03.chargedHadronIso() < absIso_cut ||
           isolationDR03.chargedHadronIso()/pf_it->p4().pt() < relIso_cut ||
           miniIso.chargedHadronIso()/pf_it->p4().pt() < miniRelIso_cut){
            outPtrP->push_back(pat::IsolatedTrack(isolationDR03, miniIso, pf_it->p4(), 
                                                  pf_it->charge(), pf_it->pdgId(), pf_it->dz(),
                                                  pf_it->dxy(), pcref));
        }

    }


    iEvent.put(std::move(outPtrP));
}


using pat::PATIsolatedTrackProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATIsolatedTrackProducer);
