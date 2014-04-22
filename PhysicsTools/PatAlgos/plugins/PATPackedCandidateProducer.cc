#include <string>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

//#define CRAZYSORT 

namespace pat {
    class PATPackedCandidateProducer : public edm::EDProducer {
        public:
            explicit PATPackedCandidateProducer(const edm::ParameterSet&);
            ~PATPackedCandidateProducer();

            virtual void produce(edm::Event&, const edm::EventSetup&);

        private:
            edm::EDGetTokenT<reco::PFCandidateCollection>    Cands_;
            edm::EDGetTokenT<reco::PFCandidateFwdPtrVector>  CandsFromPVLoose_;
            edm::EDGetTokenT<reco::PFCandidateFwdPtrVector>  CandsFromPVTight_;
            edm::EDGetTokenT<reco::VertexCollection>         PVs_;
            edm::EDGetTokenT<reco::VertexCollection>         PVOrigs_;
            edm::EDGetTokenT<reco::TrackCollection>          TKOrigs_;
            double minPtForTrackProperties_;
            // for debugging
            float calcDxy(float dx, float dy, float phi) {
                return - dx * std::sin(phi) + dy * std::cos(phi);
            }
            float calcDz(reco::Candidate::Point p, reco::Candidate::Point v, const reco::Candidate &c) {
                return p.Z()-v.Z() - ((p.X()-v.X()) * c.px() + (p.Y()-v.Y())*c.py()) * c.pz()/(c.pt()*c.pt());
            }
    };
}

pat::PATPackedCandidateProducer::PATPackedCandidateProducer(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  CandsFromPVLoose_(consumes<reco::PFCandidateFwdPtrVector>(iConfig.getParameter<edm::InputTag>("inputCollectionFromPVLoose"))),
  CandsFromPVTight_(consumes<reco::PFCandidateFwdPtrVector>(iConfig.getParameter<edm::InputTag>("inputCollectionFromPVTight"))),
  PVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("inputVertices"))),
  PVOrigs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("originalVertices"))),
  TKOrigs_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("originalTracks"))),
  minPtForTrackProperties_(iConfig.getParameter<double>("minPtForTrackProperties"))
{
  produces< std::vector<pat::PackedCandidate> > ();
  produces< edm::Association<pat::PackedCandidateCollection> > ();
  produces< edm::Association<reco::PFCandidateCollection> > ();
}

pat::PATPackedCandidateProducer::~PATPackedCandidateProducer() {}

void pat::PATPackedCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

#ifdef CRAZYSORT 
    edm::Handle<edm::View<pat::Jet> >      jets;
    iEvent.getByLabel("selectedPatJets", jets);
#endif


    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle<reco::PFCandidateFwdPtrVector> candsFromPVLoose;
    iEvent.getByToken( CandsFromPVLoose_, candsFromPVLoose );
    edm::Handle<reco::PFCandidateFwdPtrVector> candsFromPVTight;
    iEvent.getByToken( CandsFromPVTight_, candsFromPVTight );

    std::vector<pat::PackedCandidate::PVAssoc> fromPV(cands->size(), pat::PackedCandidate::NoPV);
    for (const reco::PFCandidateFwdPtr &ptr : *candsFromPVLoose) {
        if (ptr.ptr().id() == cands.id()) {
            fromPV[ptr.ptr().key()]   = pat::PackedCandidate::PVLoose;
        } else if (ptr.backPtr().id() == cands.id()) {
            fromPV[ptr.backPtr().key()] = pat::PackedCandidate::PVLoose;
        } else {
            throw cms::Exception("Configuration", "The elements from 'inputCollectionFromPVLoose' don't point to 'inputCollection'\n");
        }
    }
    for (const reco::PFCandidateFwdPtr &ptr : *candsFromPVTight) {
        if (ptr.ptr().id() == cands.id()) {
            fromPV[ptr.ptr().key()]   = pat::PackedCandidate::PVTight;
        } else if (ptr.backPtr().id() == cands.id()) {
            fromPV[ptr.backPtr().key()] = pat::PackedCandidate::PVTight;
        } else {
            throw cms::Exception("Configuration", "The elements from 'inputCollectionFromPVTight' don't point to 'inputCollection'\n");
        }
    }

    edm::Handle<reco::VertexCollection> PVOrigs;
    iEvent.getByToken( PVOrigs_, PVOrigs );
    const reco::Vertex & PVOrig = (*PVOrigs)[0];
    edm::Handle<reco::VertexCollection> PVs;
    iEvent.getByToken( PVs_, PVs );
    reco::VertexRef PV(PVs.id());
    math::XYZPoint  PVpos;
    if (!PVs->empty()) {
        PV = reco::VertexRef(PVs, 0);
        PVpos = PV->position();
    }

    edm::Handle<reco::TrackCollection> TKOrigs;
    iEvent.getByToken( TKOrigs_, TKOrigs );

    std::auto_ptr< std::vector<pat::PackedCandidate> > outPtrP( new std::vector<pat::PackedCandidate> );
    std::vector<int> mapping(cands->size());
    std::vector<int> mappingTk(TKOrigs->size(), -1);
#ifdef CRAZYSORT
    std::vector<int> jetOrder;
    std::vector<int> jetOrderReverse;
    for(unsigned int i=0;i<cands->size();i++) jetOrderReverse.push_back(-1);
    for (edm::View<pat::Jet>::const_iterator it = jets->begin(), ed = jets->end(); it != ed; ++it) {
      const  pat::Jet & jet = *it;
      const  reco::CompositePtrCandidate::daughters & dau=jet.daughterPtrVector();
      for(unsigned int  i=0;i<dau.size();i++)
	{
           if((*cands)[dau[i].key()].trackRef().isNonnull() && (*cands)[dau[i].key()].pt() > minPtForTrackProperties_){
	   jetOrder.push_back(dau[i].key());
	   jetOrderReverse[jetOrder.back()]=jetOrder.size()-1;
	   }
	}
      for(unsigned int  i=0;i<dau.size();i++)
        {
           if(!((*cands)[dau[i].key()].trackRef().isNonnull() && (*cands)[dau[i].key()].pt() > minPtForTrackProperties_)){
           jetOrder.push_back(dau[i].key());
           jetOrderReverse[jetOrder.back()]=jetOrder.size()-1;
           }
        }

    }
   for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
	if(jetOrderReverse[ic]==-1 && (*cands)[ic].trackRef().isNonnull() && (*cands)[ic].pt() > minPtForTrackProperties_)
        {
           jetOrder.push_back(ic);
           jetOrderReverse[jetOrder.back()]=jetOrder.size()-1;
        }

   }
  //all what's left
   for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        if(jetOrderReverse[ic]==-1)
        {
           jetOrder.push_back(ic);
           jetOrderReverse[jetOrder.back()]=jetOrder.size()-1;
        }

   }
#endif //CRAZYSORT


    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
#ifdef CRAZYSORT
        const reco::PFCandidate &cand=(*cands)[jetOrder[ic]];
#else
        const reco::PFCandidate &cand=(*cands)[ic];
#endif
        float phiAtVtx = cand.phi();
        const reco::Track *ctrack = 0;
        if ((abs(cand.pdgId()) == 11 || cand.pdgId() == 22) && cand.gsfTrackRef().isNonnull()) {
            ctrack = &*cand.gsfTrackRef();
        } else if (cand.trackRef().isNonnull()) {
            ctrack = &*cand.trackRef();
        }
        if (ctrack) {
            math::XYZPoint vtx = cand.vertex();
            pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
            vtx = ctrack->referencePoint();
            phiAtVtx = ctrack->phi();
            int nlost = ctrack->trackerExpectedHitsInner().numberOfLostHits();
            if (nlost == 0) { 
                if ( ctrack->hitPattern().hasValidHitInFirstPixelBarrel()) {
                    lostHits = pat::PackedCandidate::validHitInFirstPixelBarrelLayer;
                }
            } else {
                lostHits = ( nlost == 1 ? pat::PackedCandidate::oneLostInnerHit : pat::PackedCandidate::moreLostInnerHits);
            }
            outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), vtx, phiAtVtx, cand.pdgId(), PV));
          
            // properties of the best track 
            outPtrP->back().setLostInnerHits( lostHits );
	    if(outPtrP->back().pt() > minPtForTrackProperties_) {
                outPtrP->back().setTrackProperties(*ctrack);
            }

            // these things are always for the CKF track
	    if(cand.trackRef().isNonnull() && PVOrig.trackWeight(cand.trackRef()) > 0.5) {
                outPtrP->back().setFromPV(pat::PackedCandidate::PVUsedInFit);
	    } else {
                outPtrP->back().setFromPV( fromPV[ic] );
            }
            outPtrP->back().setTrackHighPurity( cand.trackRef().isNonnull() && cand.trackRef()->quality(reco::Track::highPurity) );
            if (cand.muonRef().isNonnull()) {
                outPtrP->back().setMuonID(cand.muonRef()->isStandAloneMuon(), cand.muonRef()->isGlobalMuon());
            }
        } else {
            outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), PVpos, cand.phi(), cand.pdgId(), PV));
            outPtrP->back().setFromPV( fromPV[ic] );
        }

        mapping[ic] = ic; // trivial at the moment!
        if (cand.trackRef().isNonnull() && cand.trackRef().id() == TKOrigs.id()) {
            mappingTk[cand.trackRef().key()] = ic;
        }

    }


    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put( outPtrP );

    // now build the two maps
    std::auto_ptr<edm::Association<pat::PackedCandidateCollection> > pf2pc(new edm::Association<pat::PackedCandidateCollection>(oh   ));
    std::auto_ptr<edm::Association<reco::PFCandidateCollection   > > pc2pf(new edm::Association<reco::PFCandidateCollection   >(cands));
    edm::Association<pat::PackedCandidateCollection>::Filler pf2pcFiller(*pf2pc);
    edm::Association<reco::PFCandidateCollection   >::Filler pc2pfFiller(*pc2pf);
#ifdef CRAZYSORT
    pf2pcFiller.insert(cands, jetOrderReverse.begin(), jetOrderReverse.end());
    pc2pfFiller.insert(oh   , jetOrder.begin(), jetOrder.end());
#else
    pf2pcFiller.insert(cands, mapping.begin(), mapping.end());
    pc2pfFiller.insert(oh   , mapping.begin(), mapping.end());
#endif
    // include also the mapping track -> packed PFCand
    pf2pcFiller.insert(TKOrigs, mappingTk.begin(), mappingTk.end());

    pf2pcFiller.fill();
    pc2pfFiller.fill();
    iEvent.put(pf2pc);
    iEvent.put(pc2pf);

}


using pat::PATPackedCandidateProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateProducer);
