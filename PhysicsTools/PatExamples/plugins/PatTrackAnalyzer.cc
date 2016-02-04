#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <TH1.h>
#include <TProfile.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

class PatTrackAnalyzer : public edm::EDAnalyzer  {
    public: 
	/// constructor and destructor
	PatTrackAnalyzer(const edm::ParameterSet &params);
	~PatTrackAnalyzer();

	// virtual methods called from base class EDAnalyzer
	virtual void beginJob();
	virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

    private:
	// configuration parameters
	edm::InputTag src_;
	edm::InputTag beamSpot_;

	// the list of track quality cuts to demand from the tracking
	std::vector<std::string> qualities_;

	// holder for the histograms, one set per quality flag
	struct Plots {
		TH1 *eta, *phi;
		TH1 *pt, *ptErr;
		TH1 *invPt, *invPtErr;
		TH1 *d0, *d0Err;
		TH1 *nHits;

		TProfile *pxbHitsEta, *pxeHitsEta;
		TProfile *tibHitsEta, *tobHitsEta;
		TProfile *tidHitsEta, *tecHitsEta;
	};

	std::vector<Plots> plots_;
};


PatTrackAnalyzer::PatTrackAnalyzer(const edm::ParameterSet &params) :
	src_(params.getParameter<edm::InputTag>("src")),
	beamSpot_(params.getParameter<edm::InputTag>("beamSpot")),
	qualities_(params.getParameter< std::vector<std::string> >("qualities"))
{
}

PatTrackAnalyzer::~PatTrackAnalyzer()
{
}

void PatTrackAnalyzer::beginJob()
{
	// retrieve handle to auxiliary service
	//  used for storing histograms into ROOT file
	edm::Service<TFileService> fs;

	// now book the histograms, for each category
	unsigned int nQualities = qualities_.size();

	plots_.resize(nQualities);

	for(unsigned int i = 0; i < nQualities; ++i) {
		// the name of the quality flag
		const char *quality = qualities_[i].c_str();

		// the set of plots
		Plots &plots = plots_[i];

		plots.eta = fs->make<TH1F>(Form("eta_%s", quality),
		                           Form("track \\eta (%s)", quality),
		                           100, -3, 3);
		plots.phi = fs->make<TH1F>(Form("phi_%s", quality),
		                           Form("track \\phi (%s)", quality),
		                           100, -M_PI, +M_PI);
		plots.pt = fs->make<TH1F>(Form("pt_%s", quality),
		                          Form("track p_{T} (%s)", quality),
		                          100, 0, 10);
		plots.ptErr = fs->make<TH1F>(Form("ptErr_%s", quality),
		                             Form("track p_{T} error (%s)", quality),
		                             100, 0, 1);
		plots.invPt = fs->make<TH1F>(Form("invPt_%s", quality),
		                             Form("track 1/p_{T} (%s)", quality),
		                             100, -5, 5);
		plots.invPtErr = fs->make<TH1F>(Form("invPtErr_%s", quality),
		                                Form("track 1/p_{T} error (%s)", quality),
		                                100, 0, 0.1);
		plots.d0 = fs->make<TH1F>(Form("d0_%s", quality),
		                          Form("track d0 (%s)", quality),
		                                100, 0, 0.1);
		plots.d0Err = fs->make<TH1F>(Form("d0Err_%s", quality),
		                             Form("track d0 error (%s)", quality),
		                                   100, 0, 0.1);
		plots.nHits = fs->make<TH1F>(Form("nHits_%s", quality),
		                             Form("track number of total hits (%s)", quality),
		                                   60, 0, 60);

		plots.pxbHitsEta = fs->make<TProfile>(Form("pxbHitsEta_%s", quality),
		                                      Form("#hits in Pixel Barrel (%s)", quality),
		                                           100, 0, 3);
		plots.pxeHitsEta = fs->make<TProfile>(Form("pxeHitsEta_%s", quality),
		                                      Form("#hits in Pixel Endcap (%s)", quality),
		                                           100, 0, 3);
		plots.tibHitsEta = fs->make<TProfile>(Form("tibHitsEta_%s", quality),
		                                      Form("#hits in Tracker Inner Barrel (%s)", quality),
		                                           100, 0, 3);
		plots.tobHitsEta = fs->make<TProfile>(Form("tobHitsEta_%s", quality),
		                                      Form("#hits in Tracker Outer Barrel (%s)", quality),
		                                           100, 0, 3);
		plots.tidHitsEta = fs->make<TProfile>(Form("tidHitsEta_%s", quality),
		                                      Form("#hits in Tracker Inner Disk (%s)", quality),
		                                           100, 0, 3);
		plots.tecHitsEta = fs->make<TProfile>(Form("tecHitsEta_%s", quality),
		                                      Form("#hits in Tracker Endcap (%s)", quality),
		                                           100, 0, 3);
	}
}

void PatTrackAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es)
{  
	// handles to kinds of data we might want to read
	edm::Handle<reco::BeamSpot> beamSpot;
	edm::Handle< edm::View<reco::Track> > tracksHandle;
	edm::Handle< pat::MuonCollection > muonsHandle;

	// read the beam spot
	event.getByLabel(beamSpot_, beamSpot);

	// our internal copy of track points
	// (we need this in order to able to simultaneously access tracks
	//  directly or embedded in PAT objects, like muons, normally you
	//  would iterate over the handle directly)
	std::vector<const reco::Track*> tracks;

	event.getByLabel(src_, tracksHandle);
	if (tracksHandle.isValid()) {
		// framework was able to read the collection as a view of
		// tracks, no copy them to our "tracks" variable
		for(edm::View<reco::Track>::const_iterator iter = tracksHandle->begin();
		    iter != tracksHandle->end(); ++iter)
			tracks.push_back(&*iter);
	} else {
		// does not exist or is not a track collection
		// let's assume it is a collection of PAT muons
		event.getByLabel(src_, muonsHandle);

		// and copy them over
		// NOTE: We are using ->globalTrack() here
		//       This means we are using the global fit over both
		//       the inner tracker and the muon stations!
		//       other alternatives are: innerTrack(), outerTrack()
		for(pat::MuonCollection::const_iterator iter = muonsHandle->begin();
		    iter != muonsHandle->end(); ++iter) {
		    	reco::TrackRef track = iter->globalTrack();
		    	// the muon might not be a "global" muon
		    	if (track.isNonnull())
				tracks.push_back(&*track);
		}
	}

	// we are done filling the tracks into our "tracks" vector.
	// now analyze them, once for each track quality category

	unsigned int nQualities = qualities_.size();
	for(unsigned int i = 0; i < nQualities; ++i) {
		// we convert the quality flag from its name as a string
		// to the enumeration value used by the tracking code
		// (which is essentially an integer number)
		reco::Track::TrackQuality quality = reco::Track::qualityByName(qualities_[i]);

		// our set of plots
		Plots &plots = plots_[i];

		// now loop over the tracks
		for(std::vector<const reco::Track*>::const_iterator iter = tracks.begin();
		    iter != tracks.end(); ++iter) {
			// this is our track
			const reco::Track &track = **iter;

			// ignore tracks that fail the quality cut
			if (!track.quality(quality))
				continue;

			// and fill all the plots
			plots.eta->Fill(track.eta());
			plots.phi->Fill(track.phi());

			plots.pt->Fill(track.pt());
			plots.ptErr->Fill(track.ptError());

			plots.invPt->Fill(track.qoverp());
			plots.invPtErr->Fill(track.qoverpError());

			// the transverse IP is taken with respect to
			// the beam spot instead of (0, 0)
			// because the beam spot in CMS is not at (0, 0)
			plots.d0->Fill(track.dxy(beamSpot->position()));
			plots.d0Err->Fill(track.dxyError());

			plots.nHits->Fill(track.numberOfValidHits());

			// the hit pattern contains information about
			// which modules of the detector have been hit
			const reco::HitPattern &hits = track.hitPattern();

			double absEta = std::abs(track.eta());
			// now fill the number of hits in a layer depending on eta
			plots.pxbHitsEta->Fill(absEta, hits.numberOfValidPixelBarrelHits());
			plots.pxeHitsEta->Fill(absEta, hits.numberOfValidPixelEndcapHits());
			plots.tibHitsEta->Fill(absEta, hits.numberOfValidStripTIBHits());
			plots.tobHitsEta->Fill(absEta, hits.numberOfValidStripTOBHits());
			plots.tidHitsEta->Fill(absEta, hits.numberOfValidStripTIDHits());
			plots.tecHitsEta->Fill(absEta, hits.numberOfValidStripTECHits());
		}
	}
}
	
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PatTrackAnalyzer);
