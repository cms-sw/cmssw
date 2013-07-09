#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <TH1.h>
#include <TProfile.h>

#include <Math/VectorUtil.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

class PatBJetTrackAnalyzer : public edm::EDAnalyzer  {
    public: 
	/// constructor and destructor
	PatBJetTrackAnalyzer(const edm::ParameterSet &params);
	~PatBJetTrackAnalyzer();

	// virtual methods called from base class EDAnalyzer
	virtual void beginJob();
	virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

    private:
	// configuration parameters
	edm::InputTag jets_;
	edm::InputTag tracks_;
	edm::InputTag beamSpot_;
	edm::InputTag primaryVertices_;

	double jetPtCut_;		// minimum (uncorrected) jet energy
	double jetEtaCut_;		// maximum |eta| for jet
	double maxDeltaR_;		// angle between jet and tracks

	double minPt_;			// track pt quality cut
	unsigned int minPixelHits_;	// minimum number of pixel hits
	unsigned int minTotalHits_;	// minimum number of total hits

	unsigned int nThTrack_;		// n-th hightest track to choose

	// jet flavour constants

	enum Flavour {
		ALL_JETS = 0,
		UDSG_JETS,
		C_JETS,
		B_JETS,
		NONID_JETS,
		N_JET_TYPES
	};

	TH1 *flavours_;

	// one group of plots per jet flavour;
	struct Plots {
		TH1 *allIP, *allIPErr, *allIPSig;
		TH1 *trackIP, *trackIPErr, *trackIPSig;
		TH1 *negativeIP, *negativeIPErr, *negativeIPSig;
		TH1 *nTracks, *allDeltaR;
	} plots_[N_JET_TYPES];
};

PatBJetTrackAnalyzer::PatBJetTrackAnalyzer(const edm::ParameterSet &params) :
	jets_(params.getParameter<edm::InputTag>("jets")),
	tracks_(params.getParameter<edm::InputTag>("tracks")),
	beamSpot_(params.getParameter<edm::InputTag>("beamSpot")),
	primaryVertices_(params.getParameter<edm::InputTag>("primaryVertices")),
	jetPtCut_(params.getParameter<double>("jetPtCut")),
	jetEtaCut_(params.getParameter<double>("jetEtaCut")),
	maxDeltaR_(params.getParameter<double>("maxDeltaR")),
	minPt_(params.getParameter<double>("minPt")),
	minPixelHits_(params.getParameter<unsigned int>("minPixelHits")),
	minTotalHits_(params.getParameter<unsigned int>("minTotalHits")),
	nThTrack_(params.getParameter<unsigned int>("nThTrack"))
{
}

PatBJetTrackAnalyzer::~PatBJetTrackAnalyzer()
{
}

void PatBJetTrackAnalyzer::beginJob()
{
	// retrieve handle to auxiliary service
	//  used for storing histograms into ROOT file
	edm::Service<TFileService> fs;

	flavours_ = fs->make<TH1F>("flavours", "jet flavours", 5, 0, 5);

	// book histograms for all jet flavours
	for(unsigned int i = 0; i < N_JET_TYPES; i++) {
		Plots &plots = plots_[i];
		const char *flavour, *name;

		switch((Flavour)i) {
		    case ALL_JETS:
			flavour = "all jets";
			name = "all";
			break;
		    case UDSG_JETS:
			flavour = "light flavour jets";
			name = "udsg";
			break;
		    case C_JETS:
			flavour = "charm jets";
			name = "c";
			break;
		    case B_JETS:
			flavour = "bottom jets";
			name = "b";
			break;
		    default:
			flavour = "unidentified jets";
			name = "ni";
			break;
		}

		plots.allIP = fs->make<TH1F>(Form("allIP_%s", name),
		                             Form("signed IP for all tracks in %s", flavour),
		                             100, -0.1, 0.2);
		plots.allIPErr = fs->make<TH1F>(Form("allIPErr_%s", name),
		                                Form("error of signed IP for all tracks in %s", flavour),
		                                100, 0, 0.05);
		plots.allIPSig = fs->make<TH1F>(Form("allIPSig_%s", name),
		                                Form("signed IP significance for all tracks in %s", flavour),
		                                100, -10, 20);

		plots.trackIP = fs->make<TH1F>(Form("trackIP_%s", name),
		                               Form("signed IP for selected positive track in %s", flavour),
		                               100, -0.1, 0.2);
		plots.trackIPErr = fs->make<TH1F>(Form("trackIPErr_%s", name),
		                                  Form("error of signed IP for selected positive track in %s", flavour),
		                                  100, 0, 0.05);
		plots.trackIPSig = fs->make<TH1F>(Form("trackIPSig_%s", name),
		                                  Form("signed IP significance for selected positive track in %s", flavour),
		                                  100, -10, 20);

		plots.negativeIP = fs->make<TH1F>(Form("negativeIP_%s", name),
		                                  Form("signed IP for selected negative track in %s", flavour),
		                                  100, -0.2, 0.1);
		plots.negativeIPErr = fs->make<TH1F>(Form("negativeIPErr_%s", name),
		                                     Form("error of signed IP for selected negative track in %s", flavour),
		                                     100, 0, 0.05);
		plots.negativeIPSig = fs->make<TH1F>(Form("negativeIPSig_%s", name),
		                                     Form("signed IP significance for selected negative track in %s", flavour),
		                                     100, -20, 10);

		plots.nTracks = fs->make<TH1F>(Form("nTracks_%s", name),
		                               Form("number of usable tracks in %s", flavour),
		                               30, 0, 30);
		plots.allDeltaR = fs->make<TH1F>(Form("allDeltaR_%s", name),
		                                 Form("\\DeltaR between track and %s", flavour),
		                                 100, 0, 1);
	}
}

// helper function to sort the tracks by impact parameter significance

static bool significanceHigher(const Measurement1D &meas1,
                               const Measurement1D &meas2)
{ return meas1.significance() > meas2.significance(); }

void PatBJetTrackAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es)
{  
	// handle to the primary vertex collection
	edm::Handle<reco::VertexCollection> pvHandle;
	event.getByLabel(primaryVertices_, pvHandle);

	// handle to the tracks collection
	edm::Handle<reco::TrackCollection> tracksHandle;
	event.getByLabel(tracks_, tracksHandle);

	// handle to the jets collection
	edm::Handle<pat::JetCollection> jetsHandle;
	event.getByLabel(jets_, jetsHandle);

	// handle to the beam spot
	edm::Handle<reco::BeamSpot> beamSpot;
	event.getByLabel(beamSpot_, beamSpot);

	// rare case of no reconstructed primary vertex
	if (pvHandle->empty())
		return;

	// extract the position of the (most probable) reconstructed vertex
	math::XYZPoint pv = (*pvHandle)[0].position();

	// now go through all jets
	for(pat::JetCollection::const_iterator jet = jetsHandle->begin();
	    jet != jetsHandle->end(); ++jet) {

		// only look at jets that pass the pt and eta cut
		if (jet->pt() < jetPtCut_ ||
		    std::abs(jet->eta()) > jetEtaCut_)
			continue;

		Flavour flavour;
		// find out the jet flavour (differs between quark and anti-quark)
		switch(std::abs(jet->partonFlavour())) {
		    case 1:
		    case 2:
		    case 3:
		    case 21:
			flavour = UDSG_JETS;
			break;
		    case 4:
			flavour = C_JETS;
			break;
		    case 5:
			flavour = B_JETS;
			break;
		    default:
			flavour = NONID_JETS;
		}
	
		// simply count the number of accepted jets
		flavours_->Fill(ALL_JETS);
		flavours_->Fill(flavour);

		// this vector will contain IP value / error pairs
		std::vector<Measurement1D> ipValErr;

		// Note: PAT is also able to store associated tracks
		//       within the jet object, so we don't have to do the
		//       matching ourselves
		// (see ->associatedTracks() method)
		// However, using this we can't play with the DeltaR cone
		// withour rerunning the PAT producer

		// now loop through all tracks
		for(reco::TrackCollection::const_iterator track = tracksHandle->begin();
		    track != tracksHandle->end(); ++track) {

			// check the quality criteria
			if (track->pt() < minPt_ ||
			    track->hitPattern().numberOfValidHits() < (int)minTotalHits_ ||
			    track->hitPattern().numberOfValidPixelHits() < (int)minPixelHits_)
				continue;

			// check the Delta R between jet axis and track
			// (Delta_R^2 = Delta_Eta^2 + Delta_Phi^2)
			double deltaR = ROOT::Math::VectorUtil::DeltaR(
					jet->momentum(), track->momentum());

			plots_[ALL_JETS].allDeltaR->Fill(deltaR);
			plots_[flavour].allDeltaR->Fill(deltaR);

			// only look at tracks in jet cone
			if (deltaR > maxDeltaR_)
				continue;

			// What follows here is an approximation!
			//
			// The dxy() method of the tracks does a linear
			// extrapolation from the track reference position
			// given as the closest point to the beam spot
			// with respect to the given vertex.
			// Since we are using primary vertices, this
			// approximation works well
			//
			// In order to get better results, the
			// "TransientTrack" and specialised methods have
			// to be used.
			// Or look at the "impactParameterTagInfos",
			// which contains the precomputed information
			// from the official b-tagging algorithms
			//
			// see ->tagInfoTrackIP() method

			double ipError = track->dxyError();
			double ipValue = std::abs(track->dxy(pv));

			// in order to compute the sign, we check if
			// the point of closest approach to the vertex
			// is in front or behind the vertex.
			// Again, we a linear approximation
			// 
			// dot product between reference point and jet axis

			math::XYZVector closestPoint = track->referencePoint() - beamSpot->position();
			// only interested in transverse component, z -> 0
			closestPoint.SetZ(0.);
			double sign = closestPoint.Dot(jet->momentum());

			if (sign < 0)
				ipValue = -ipValue;

			ipValErr.push_back(Measurement1D(ipValue, ipError));
		}

		// now order all tracks by significance (highest first)
		std::sort(ipValErr.begin(), ipValErr.end(), significanceHigher);

		plots_[ALL_JETS].nTracks->Fill(ipValErr.size());
		plots_[flavour].nTracks->Fill(ipValErr.size());

		// plot all tracks

		for(std::vector<Measurement1D>::const_iterator iter = ipValErr.begin();
		    iter != ipValErr.end(); ++iter) {
			plots_[ALL_JETS].allIP->Fill(iter->value());
			plots_[flavour].allIP->Fill(iter->value());

			plots_[ALL_JETS].allIPErr->Fill(iter->error());
			plots_[flavour].allIPErr->Fill(iter->error());

			// significance (is really just value / error)
			plots_[ALL_JETS].allIPSig->Fill(iter->significance());
			plots_[flavour].allIPSig->Fill(iter->significance());
		}

		// check if we have enough tracks to fulfill the
		// n-th track requirement
		if (ipValErr.size() < nThTrack_)
			continue;

		// pick the n-th highest track
		const Measurement1D *trk = &ipValErr[nThTrack_ - 1];

		plots_[ALL_JETS].trackIP->Fill(trk->value());
		plots_[flavour].trackIP->Fill(trk->value());

		plots_[ALL_JETS].trackIPErr->Fill(trk->error());
		plots_[flavour].trackIPErr->Fill(trk->error());

		plots_[ALL_JETS].trackIPSig->Fill(trk->significance());
		plots_[flavour].trackIPSig->Fill(trk->significance());

		// here we define a "negative tagger", i.e. we take
		// the track with the n-lowest signed IP
		// (i.e. preferrably select tracks that appear to become
		//  from "behind" the primary vertex, supposedly mismeasured
		//  tracks really coming from the primary vertex, and
		//  the contamination of displaced tracks should be small)
		trk = &ipValErr[ipValErr.size() - nThTrack_];

		plots_[ALL_JETS].negativeIP->Fill(trk->value());
		plots_[flavour].negativeIP->Fill(trk->value());

		plots_[ALL_JETS].negativeIPErr->Fill(trk->error());
		plots_[flavour].negativeIPErr->Fill(trk->error());

		plots_[ALL_JETS].negativeIPSig->Fill(trk->significance());
		plots_[flavour].negativeIPSig->Fill(trk->significance());
	}
}
	
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PatBJetTrackAnalyzer);
