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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class PatVertexAnalyzer : public edm::EDAnalyzer  {
    public: 
	/// constructor and destructor
	PatVertexAnalyzer(const edm::ParameterSet &params);
	~PatVertexAnalyzer();

	// virtual methods called from base class EDAnalyzer
	virtual void beginJob();
	virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

    private:
	// configuration parameters
	edm::InputTag src_;
	edm::InputTag genParticles_;

	TH1 *nVertices_, *nTracks_;
	TH1 *x_, *y_, *z_;
	TH1 *xErr_, *yErr_, *zErr_;
	TH1 *xDelta_, *yDelta_, *zDelta_;
	TH1 *xPull_, *yPull_, *zPull_;
};

PatVertexAnalyzer::PatVertexAnalyzer(const edm::ParameterSet &params) :
	src_(params.getParameter<edm::InputTag>("src")),
	genParticles_(params.getParameter<edm::InputTag>("mc"))
{
}

PatVertexAnalyzer::~PatVertexAnalyzer()
{
}

void PatVertexAnalyzer::beginJob()
{
	// retrieve handle to auxiliary service
	//  used for storing histograms into ROOT file
	edm::Service<TFileService> fs;

	nVertices_ = fs->make<TH1F>("nVertices", "number of reconstructed primary vertices", 50, 0, 50);
	nTracks_ = fs->make<TH1F>("nTracks", "number of tracks at primary vertex", 100, 0, 300);
	x_ = fs->make<TH1F>("pvX", "primary vertex x", 100, -0.1, 0.1);
	y_ = fs->make<TH1F>("pvY", "primary vertex y", 100, -0.1, 0.1);
	z_ = fs->make<TH1F>("pvZ", "primary vertex z", 100, -30, 30);
	xErr_ = fs->make<TH1F>("pvErrorX", "primary vertex x error", 100, 0, 0.005);
	yErr_ = fs->make<TH1F>("pvErrorY", "primary vertex y error", 100, 0, 0.005);
	zErr_ = fs->make<TH1F>("pvErrorZ", "primary vertex z error", 100, 0, 0.01);
	xDelta_ = fs->make<TH1F>("pvDeltaX", "x shift wrt simulated vertex", 100, -0.01, 0.01);
	yDelta_ = fs->make<TH1F>("pvDeltaY", "y shift wrt simulated vertex", 100, -0.01, 0.01);
	zDelta_ = fs->make<TH1F>("pvDeltaZ", "z shift wrt simulated vertex", 100, -0.02, 0.02);
	xPull_ = fs->make<TH1F>("pvPullX", "primary vertex x pull", 100, -5, 5);
	yPull_ = fs->make<TH1F>("pvPullY", "primary vertex y pull", 100, -5, 5);
	zPull_ = fs->make<TH1F>("pvPullZ", "primary vertex z pull", 100, -5, 5);
}

void PatVertexAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es)
{  
	// handle to the primary vertex collection
	edm::Handle<reco::VertexCollection> pvHandle;
	event.getByLabel(src_, pvHandle);

	// handle to the generator particles (i.e. the MC truth)
	edm::Handle<reco::GenParticleCollection> genParticlesHandle;
	event.getByLabel(genParticles_, genParticlesHandle);

	// extract the position of the simulated vertex
	math::XYZPoint simPV = (*genParticlesHandle)[2].vertex();

	// the number of reconstructed primary vertices
	nVertices_->Fill(pvHandle->size());

	// if we have at least one, use the first (highest pt^2 sum)
	if (!pvHandle->empty()) {
		const reco::Vertex &pv = (*pvHandle)[0];

		nTracks_->Fill(pv.tracksSize());

		x_->Fill(pv.x());
		y_->Fill(pv.y());
		z_->Fill(pv.z());

		xErr_->Fill(pv.xError());
		yErr_->Fill(pv.yError());
		zErr_->Fill(pv.zError());

		xDelta_->Fill(pv.x() - simPV.X());
		yDelta_->Fill(pv.y() - simPV.Y());
		zDelta_->Fill(pv.z() - simPV.Z());

		xPull_->Fill((pv.x() - simPV.X()) / pv.xError());
		yPull_->Fill((pv.y() - simPV.Y()) / pv.yError());
		zPull_->Fill((pv.z() - simPV.Z()) / pv.zError());

		// we could access the tracks using the
		// pv.tracks_begin() ... pv.tracks_end() iterators
	}
}
	
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PatVertexAnalyzer);
