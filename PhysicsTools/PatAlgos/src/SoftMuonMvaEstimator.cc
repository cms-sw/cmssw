#include "PhysicsTools/PatAlgos/interface/SoftMuonMvaEstimator.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace pat;

SoftMuonMvaEstimator::SoftMuonMvaEstimator():
	tmvaReader_("!Color:!Silent:Error"),
	initialized_(false),
	mva_(0)
{}

void SoftMuonMvaEstimator::initialize(std::string weightsfile)
{
	if (initialized_) return;
	tmvaReader_.AddVariable("segComp",			&segmentCompatibility_);
	tmvaReader_.AddVariable("chi2LocMom",			&chi2LocalMomentum_);
	tmvaReader_.AddVariable("chi2LocPos",			&chi2LocalPosition_);
	tmvaReader_.AddVariable("glbTrackTailProb",		&glbTrackProbability_);
	tmvaReader_.AddVariable("iValFrac",			&iValidFraction_);
	tmvaReader_.AddVariable("LWH",				&layersWithMeasurement_);
	tmvaReader_.AddVariable("kinkFinder",			&trkKink_);
	tmvaReader_.AddVariable("TMath::Log(2+glbKinkFinder)",	&log2PlusGlbKink_);
	tmvaReader_.AddVariable("timeAtIpInOutErr",		&timeAtIpInOutErr_);
	tmvaReader_.AddVariable("outerChi2",			&outerChi2_);
	tmvaReader_.AddVariable("innerChi2",			&innerChi2_);
	tmvaReader_.AddVariable("trkRelChi2",			&trkRelChi2_);
	tmvaReader_.AddVariable("vMuonHitComb",			&vMuonHitComb_);
	tmvaReader_.AddVariable("Qprod",			&qProd_);

	tmvaReader_.AddSpectator("pID",			       	&pID_);
	tmvaReader_.AddSpectator("pt",				&pt_);
	tmvaReader_.AddSpectator("eta",				&eta_);
	tmvaReader_.AddSpectator("MomID",			&momID_);

	tmvaReader_.BookMVA("BDT", weightsfile);
	initialized_ = true;	
}

void SoftMuonMvaEstimator::computeMva(const pat::Muon& muon)
{
	if (not initialized_)
		throw cms::Exception("FatalError") << "SoftMuonMVA is not initialized";

	reco::TrackRef gTrack = muon.globalTrack();
	reco::TrackRef iTrack = muon.innerTrack();
	reco::TrackRef oTrack = muon.outerTrack();

	if(!(muon.innerTrack().isNonnull() and 
	     muon.outerTrack().isNonnull() and 
	     muon.globalTrack().isNonnull()))
	{
	  mva_ = -1; 
	  return;
	}

	//VARIABLE EXTRACTION
	pt_ = muon.pt();
	eta_ = muon.eta();
	dummy_ = -1;
	momID_ = dummy_;
	pID_ = dummy_;


	chi2LocalMomentum_ = muon.combinedQuality().chi2LocalMomentum;
	chi2LocalPosition_ = muon.combinedQuality().chi2LocalPosition;
	glbTrackProbability_ =  muon.combinedQuality().glbTrackProbability;
	trkRelChi2_ =  muon.combinedQuality().trkRelChi2;

	trkKink_ = muon.combinedQuality().trkKink;
	log2PlusGlbKink_ =  TMath::Log(2+muon.combinedQuality().glbKink);
	segmentCompatibility_ = muon.segmentCompatibility();

	timeAtIpInOutErr_ = muon.time().timeAtIpInOutErr;

	//TRACK RELATED VARIABLES
	
	iValidFraction_ =  iTrack->validFraction();
	innerChi2_ =  iTrack->normalizedChi2();
	layersWithMeasurement_ = iTrack->hitPattern().trackerLayersWithMeasurement();

	outerChi2_ = oTrack->normalizedChi2();

	qProd_ =  iTrack->charge()*oTrack->charge();

	//vComb Calculation

	const reco::HitPattern &gMpattern = gTrack->hitPattern();

	std::vector<int> fvDThits = {0,0,0,0};
	std::vector<int> fvRPChits = {0,0,0,0};
	std::vector<int> fvCSChits = {0,0,0,0};

	vMuonHitComb_ = 0;

	for (int i=0;i<gMpattern.numberOfAllHits(reco::HitPattern::TRACK_HITS);i++){

	  uint32_t hit = gMpattern.getHitPattern(reco::HitPattern::TRACK_HITS, i);
	  if ( !gMpattern.validHitFilter(hit) ) continue;

	  int muStation0 = gMpattern.getMuonStation(hit) - 1;
	  if (muStation0 >=0 && muStation0 < 4){
	    if (gMpattern.muonDTHitFilter(hit)) fvDThits[muStation0]++;
	    if (gMpattern.muonRPCHitFilter(hit)) fvRPChits[muStation0]++;
	    if (gMpattern.muonCSCHitFilter(hit)) fvCSChits[muStation0]++;
	  }

	}
  

	for (unsigned int station = 0; station < 4; ++station) {

	  vMuonHitComb_ += (fvDThits[station])/2.;
	  vMuonHitComb_ += fvRPChits[station];

	  if (fvCSChits[station] > 6){
	    vMuonHitComb_ += 6; 
	  }else{
	    vMuonHitComb_ += fvCSChits[station];
	  }

	}

	if(chi2LocalMomentum_ < 5000 and chi2LocalPosition_ < 2000 and
	   glbTrackProbability_ < 5000 and trkKink_ < 900 and
	   log2PlusGlbKink_ < 50 and timeAtIpInOutErr_ < 4 and
	   outerChi2_ < 1000 and innerChi2_ < 10 and trkRelChi2_ < 3) 
	{
	  mva_ = tmvaReader_.EvaluateMVA("BDT");
	}else{ mva_ = -1; }

}
