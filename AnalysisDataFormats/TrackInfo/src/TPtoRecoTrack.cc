#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrack.h"


// Constructors
TPtoRecoTrack::TPtoRecoTrack() 
{
	SetBeamSpot(math::XYZPoint(-9999.0, -9999.0, -9999.0));	
	SetTrackingParticlePCA(GlobalPoint(-9999.0, -9999.0, -9999.0));
   	SetTrackingParticleMomentumPCA(GlobalVector(-9999.0, -9999.0, -9999.0));
}


TPtoRecoTrack::~TPtoRecoTrack()
{
}

TrackingParticle TPtoRecoTrack::TPMother(unsigned short i)
{
	TrackingParticleRefVector mothers = TP().parentVertex()->sourceTracks();
	//if(mothers.size()) return *(*mothers.begin());
	//return i < mothers.size() ? *mothers[i] : TrackingParticle();
	TrackingParticle tp;
	tp.setPdgId(-9999);
	return i < mothers.size() ? *(mothers[i]) : tp;
/*
	HepMC::GenParticle *gp = GetTrackingParticle().genParticle().size() > 0 ? *GetTrackingParticle().genParticle_begin() : HepMC::GenParticle();
	
	for( GenVertex::particles_in_const_iterator i = gp->production_vertex()->particles_in_const_begin(); i != h->production_vertex()->particles_in_const_end(); i++)
	{
*/		
}

int TPtoRecoTrack::numTPMothers2()
{
	return GetTrackingParticle().genParticle_begin() != GetTrackingParticle().genParticle_end() ? (*GetTrackingParticle().genParticle_begin())->production_vertex()->particles_in_size() : 0;
}


