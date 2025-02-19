#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <TH1.h>
#include <TProfile.h>

#include <Math/VectorUtil.h>
#include <Math/GenVector/PxPyPzE4D.h>
#include <Math/GenVector/PxPyPzM4D.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

class PatBJetVertexAnalyzer : public edm::EDAnalyzer  {
    public: 
	/// constructor and destructor
	PatBJetVertexAnalyzer(const edm::ParameterSet &params);
	~PatBJetVertexAnalyzer();

	// virtual methods called from base class EDAnalyzer
	virtual void beginJob();
	virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

    private:
	// configuration parameters
	edm::InputTag jets_;

	double jetPtCut_;		// minimum (uncorrected) jet energy
	double jetEtaCut_;		// maximum |eta| for jet
	double maxDeltaR_;		// angle between jet and tracks

	// jet flavour constants

	enum Flavour {
		ALL_JETS = 0,
		UDSG_JETS,
		C_JETS,
		B_JETS,
		NONID_JETS,
		N_JET_TYPES
	};

	TH1 * flavours_;

	// one group of plots per jet flavour;
	struct Plots {
		TH1 *nVertices;
		TH1 *deltaR, *mass, *dist, *distErr, *distSig;
		TH1 *nTracks, *chi2;
	} plots_[N_JET_TYPES];
};

PatBJetVertexAnalyzer::PatBJetVertexAnalyzer(const edm::ParameterSet &params) :
	jets_(params.getParameter<edm::InputTag>("jets")),
	jetPtCut_(params.getParameter<double>("jetPtCut")),
	jetEtaCut_(params.getParameter<double>("jetEtaCut"))
{
}

PatBJetVertexAnalyzer::~PatBJetVertexAnalyzer()
{
}

void PatBJetVertexAnalyzer::beginJob()
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

		plots.nVertices = fs->make<TH1F>(Form("nVertices_%s", name),
		                                 Form("number of secondary vertices in %s", flavour),
		                                 5, 0, 5);
		plots.deltaR = fs->make<TH1F>(Form("deltaR_%s", name),
		                              Form("\\DeltaR between vertex direction and jet direction in %s", flavour),
		                              100, 0, 0.5);
		plots.mass = fs->make<TH1F>(Form("mass_%s", name),
		                            Form("vertex mass in %s", flavour),
		                            100, 0, 10);
		plots.dist = fs->make<TH1F>(Form("dist_%s", name),
		                            Form("Transverse distance between PV and SV in %s", flavour),
		                            100, 0, 2);
		plots.distErr = fs->make<TH1F>(Form("distErr_%s", name),
		                               Form("Transverse distance error between PV and SV in %s", flavour),
		                               100, 0, 0.5);
		plots.distSig = fs->make<TH1F>(Form("distSig_%s", name),
		                               Form("Transverse distance significance between PV and SV in %s", flavour),
		                               100, 0, 50);
		plots.nTracks = fs->make<TH1F>(Form("nTracks_%s", name),
		                               Form("number of tracks at secondary vertex in %s", flavour),
		                               20, 0, 20);
		plots.chi2 = fs->make<TH1F>(Form("chi2_%s", name),
		                            Form("secondary vertex fit \\chi^{2} in %s", flavour),
		                            100, 0, 50);
	}
}

// helper function to sort the tracks by impact parameter significance

void PatBJetVertexAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es)
{  
	// handle to the jets collection
	edm::Handle<pat::JetCollection> jetsHandle;
	event.getByLabel(jets_, jetsHandle);

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

		// retrieve the "secondary vertex tag infos"
		// this is the output of the b-tagging reconstruction code
		// and contains secondary vertices in the jets
		const reco::SecondaryVertexTagInfo &svTagInfo =
					*jet->tagInfoSecondaryVertex();

		// count the number of secondary vertices
		plots_[ALL_JETS].nVertices->Fill(svTagInfo.nVertices());
		plots_[flavour].nVertices->Fill(svTagInfo.nVertices());

		// ignore jets without SV from now on
		if (svTagInfo.nVertices() < 1)
			continue;

		// pick the first secondary vertex (the "best" one)
		const reco::Vertex &sv = svTagInfo.secondaryVertex(0);

		// and plot number of tracks and chi^2
		plots_[ALL_JETS].nTracks->Fill(sv.tracksSize());
		plots_[flavour].nTracks->Fill(sv.tracksSize());

		plots_[ALL_JETS].chi2->Fill(sv.chi2());
		plots_[flavour].chi2->Fill(sv.chi2());

		// the precomputed transverse distance to the primary vertex
		Measurement1D distance = svTagInfo.flightDistance(0, true);

		plots_[ALL_JETS].dist->Fill(distance.value());
		plots_[flavour].dist->Fill(distance.value());

		plots_[ALL_JETS].distErr->Fill(distance.error());
		plots_[flavour].distErr->Fill(distance.error());

		plots_[ALL_JETS].distSig->Fill(distance.significance());
		plots_[flavour].distSig->Fill(distance.significance());


		// the precomputed direction with respect to the primary vertex
		GlobalVector dir = svTagInfo.flightDirection(0);

		// unfortunately CMSSW hsa all kinds of vectors,
		// and sometimes we need to convert them *sigh*
		math::XYZVector dir2(dir.x(), dir.y(), dir.z());

		// compute a few variables that we are plotting
		double deltaR = ROOT::Math::VectorUtil::DeltaR(
						jet->momentum(), dir2);

		plots_[ALL_JETS].deltaR->Fill(deltaR);
		plots_[flavour].deltaR->Fill(deltaR);

		// compute the invariant mass from a four-vector sum
		math::XYZTLorentzVector trackFourVectorSum;

		// loop over all tracks in the vertex
		for(reco::Vertex::trackRef_iterator track = sv.tracks_begin();
		    track != sv.tracks_end(); ++track) {
			ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec;
			vec.SetPx((*track)->px());
			vec.SetPy((*track)->py());
			vec.SetPz((*track)->pz());
			vec.SetM(0.13957);	// pion mass
			trackFourVectorSum += vec;
		}

		// get the invariant mass: sqrt(E² - px² - py² - pz²)
		double vertexMass = trackFourVectorSum.M();

		plots_[ALL_JETS].mass->Fill(vertexMass);
		plots_[flavour].mass->Fill(vertexMass);
	}
}
	
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PatBJetVertexAnalyzer);
