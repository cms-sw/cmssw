#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class LowPtGSFToPackedCandidateLinker : public edm::global::EDProducer<> {
public:
	explicit LowPtGSFToPackedCandidateLinker(const edm::ParameterSet&);
	~LowPtGSFToPackedCandidateLinker() override;
    
	void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
	static void fillDescriptions(edm::ConfigurationDescriptions&);
   
private:
	const edm::EDGetTokenT<reco::PFCandidateCollection> pfcands_;
	const edm::EDGetTokenT<pat::PackedCandidateCollection> packed_;
	const edm::EDGetTokenT<pat::PackedCandidateCollection> lost_tracks_;
	const edm::EDGetTokenT<reco::TrackCollection>          tracks_;
	const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > pf2packed_;
	const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > lost2trk_;
	const edm::EDGetTokenT< edm::Association<reco::TrackCollection> > gsf2trk_;	
	const edm::EDGetTokenT< std::vector<reco::GsfTrack> > gsftracks_;	
};

LowPtGSFToPackedCandidateLinker::LowPtGSFToPackedCandidateLinker(const edm::ParameterSet& iConfig) :
  pfcands_{consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates"))},
  packed_{consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedCandidates"))},
  lost_tracks_{consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostTracks"))},
  tracks_{consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))},
  pf2packed_{consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedCandidates"))},
  lost2trk_{consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("lostTracks"))},
  gsf2trk_{consumes<edm::Association<reco::TrackCollection> >(iConfig.getParameter<edm::InputTag>("gsfToTrack"))},
  gsftracks_{consumes<std::vector<reco::GsfTrack> >(iConfig.getParameter<edm::InputTag>("gsfTracks"))} {     
		produces< edm::Association<pat::PackedCandidateCollection> > ("packedCandidates");
		produces< edm::Association<pat::PackedCandidateCollection> > ("lostTracks");
	}

LowPtGSFToPackedCandidateLinker::~LowPtGSFToPackedCandidateLinker() {}

void LowPtGSFToPackedCandidateLinker::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
	edm::Handle<reco::PFCandidateCollection> pfcands;
	iEvent.getByToken(pfcands_, pfcands);

	edm::Handle<pat::PackedCandidateCollection> packed;
	iEvent.getByToken(packed_, packed);

	edm::Handle<pat::PackedCandidateCollection> lost_tracks;
	iEvent.getByToken(lost_tracks_, lost_tracks);

	edm::Handle<edm::Association<pat::PackedCandidateCollection> > pf2packed;
	iEvent.getByToken(pf2packed_, pf2packed);

	edm::Handle<edm::Association<pat::PackedCandidateCollection> > lost2trk_assoc;
	iEvent.getByToken(lost2trk_, lost2trk_assoc);

	edm::Handle<std::vector<reco::GsfTrack> > gsftracks;
	iEvent.getByToken(gsftracks_, gsftracks);

	edm::Handle<reco::TrackCollection> tracks;
	iEvent.getByToken(tracks_, tracks);

	edm::Handle<edm::Association<reco::TrackCollection> > gsf2trk;
	iEvent.getByToken(gsf2trk_, gsf2trk);

	// collection sizes, for reference
	const size_t npf = pfcands->size();
	const size_t npacked = packed->size();
	const size_t nlost = lost_tracks->size();
	const size_t ntracks = tracks->size();
	const size_t ngsf = gsftracks->size();

	//store index mapping in vectors for easy and fast access
	std::vector<size_t> trk2packed(ntracks, npacked);
	std::vector<size_t> trk2lost(ntracks, nlost);

	//store auxiliary mappings for association
	std::vector<int> gsf2pack(ngsf, -1);
	std::vector<int> gsf2lost(ngsf, -1);

	//electrons will never store their track (they store the Gsf track)
	//map PackedPF <--> Track
	for(unsigned int icand=0; icand < npf; ++icand) {
		edm::Ref<reco::PFCandidateCollection> pf_ref(pfcands,icand);
		const reco::PFCandidate &cand = pfcands->at(icand); 
		auto packed_ref = (*pf2packed)[pf_ref];
		if(cand.charge() && packed_ref.isNonnull() && cand.trackRef().isNonnull() 
			 && cand.trackRef().id() == tracks.id() ) { 
			size_t trkid = cand.trackRef().index();
			size_t packid = packed_ref.index();
			trk2packed[trkid] = packid;
		}
	}

	//map LostTrack <--> Track
	for(unsigned int itrk=0; itrk < ntracks; ++itrk) {
		reco::TrackRef key(tracks, itrk);
		pat::PackedCandidateRef lostTrack = (*lost2trk_assoc)[key];
		if(lostTrack.isNonnull()) {
			size_t ilost = lostTrack.index(); //assumes that LostTracks are all made from the same track collection
			trk2lost[itrk] = ilost;
		}
	}	

	//map Track --> GSF and fill GSF --> PackedCandidates and GSF --> Lost associations
	for(unsigned int igsf=0; igsf < ngsf; ++igsf) {
		reco::GsfTrackRef gref(gsftracks, igsf);
		reco::TrackRef trk = (*gsf2trk)[gref];
		if(trk.id() != tracks.id()) {
			throw cms::Exception("WrongCollection", "The reco::Track collection used to match against the GSF Tracks was not used to produce such tracks");
		}
		size_t trkid = trk.index();

		if(trk2packed[trkid] != npacked) {
			gsf2pack[igsf] = trk2packed[trkid];
		} 
		if(trk2lost[trkid] != nlost) {
			gsf2lost[igsf] = trk2lost[trkid];
		}
	}

	// create output collections from the mappings
	auto assoc_gsf2pack = std::make_unique< edm::Association<pat::PackedCandidateCollection> >(packed);
	edm::Association<pat::PackedCandidateCollection>::Filler gsf2pack_filler(*assoc_gsf2pack);
	gsf2pack_filler.insert(gsftracks, gsf2pack.begin(), gsf2pack.end());
	gsf2pack_filler.fill(); 
	iEvent.put(std::move(assoc_gsf2pack), "packedCandidates");   

	auto assoc_gsf2lost = std::make_unique< edm::Association<pat::PackedCandidateCollection> >(lost_tracks);
	edm::Association<pat::PackedCandidateCollection>::Filler gsf2lost_filler(*assoc_gsf2lost);
	gsf2lost_filler.insert(gsftracks, gsf2lost.begin(), gsf2lost.end());
	gsf2lost_filler.fill();
	iEvent.put(std::move(assoc_gsf2lost), "lostTracks");
}

void LowPtGSFToPackedCandidateLinker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	edm::ParameterSetDescription desc;
	desc.add<edm::InputTag>("PFCandidates", edm::InputTag("particleFlow"));
	desc.add<edm::InputTag>("packedCandidates", edm::InputTag("packedPFCandidates"));
	desc.add<edm::InputTag>("lostTracks", edm::InputTag("lostTracks"));
	desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
	desc.add<edm::InputTag>("gsfToTrack", edm::InputTag("lowPtGsfToTrackLinks"));
	desc.add<edm::InputTag>("gsfTracks", edm::InputTag("lowPtGsfEleGsfTracks"));
	descriptions.add("lowPtGsfLinksDefault", desc);
}

DEFINE_FWK_MODULE(LowPtGSFToPackedCandidateLinker);
