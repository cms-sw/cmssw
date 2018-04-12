#include "PhysicsTools/PatAlgos/interface/SoftMuonMvaEstimator.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TMVA/Reader.h"
#include "TMVA/MethodBDT.h"

using namespace pat;

namespace {
  constexpr char softmuon_mva_name[] = "BDT";
}

SoftMuonMvaEstimator::SoftMuonMvaEstimator(const std::string& weightsfile)
{
  TMVA::Reader tmvaReader("!Color:!Silent:Error");
  tmvaReader.AddVariable("segComp",			&segmentCompatibility_);
  tmvaReader.AddVariable("chi2LocMom",			&chi2LocalMomentum_);
  tmvaReader.AddVariable("chi2LocPos",			&chi2LocalPosition_);
  tmvaReader.AddVariable("glbTrackTailProb",		&glbTrackProbability_);
  tmvaReader.AddVariable("iValFrac",			&iValidFraction_);
  tmvaReader.AddVariable("LWH",				&layersWithMeasurement_);
  tmvaReader.AddVariable("kinkFinder",			&trkKink_);
  tmvaReader.AddVariable("TMath::Log(2+glbKinkFinder)",	&log2PlusGlbKink_);
  tmvaReader.AddVariable("timeAtIpInOutErr",		&timeAtIpInOutErr_);
  tmvaReader.AddVariable("outerChi2",			&outerChi2_);
  tmvaReader.AddVariable("innerChi2",			&innerChi2_);
  tmvaReader.AddVariable("trkRelChi2",			&trkRelChi2_);
  tmvaReader.AddVariable("vMuonHitComb",		&vMuonHitComb_);
  tmvaReader.AddVariable("Qprod",			&qProd_);

  tmvaReader.AddSpectator("pID",			&pID_);
  tmvaReader.AddSpectator("pt",				&pt_);
  tmvaReader.AddSpectator("eta",			&eta_);
  tmvaReader.AddSpectator("MomID",			&momID_);

  std::unique_ptr<TMVA::IMethod> temp( tmvaReader.BookMVA(softmuon_mva_name, weightsfile.c_str()) );
  gbrForest_.reset(new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmvaReader.FindMVA(softmuon_mva_name) ) ) );
}

SoftMuonMvaEstimator::~SoftMuonMvaEstimator() { }

namespace {
  enum inputIndexes {
    kSegmentCompatibility,
    kChi2LocalMomentum,
    kChi2LocalPosition,
    kGlbTrackProbability,
    kIValidFraction,
    kLayersWithMeasurement,
    kTrkKink,
    kLog2PlusGlbKink,
    kTimeAtIpInOutErr,
    kOuterChi2,
    kInnerChi2,
    kTrkRelChi2,
    kVMuonHitComb,
    kQProd,
    kPID,
    kPt,
    kEta,
    kMomID,
    kLast
  };
}

float SoftMuonMvaEstimator::computeMva(const pat::Muon& muon) const
{
	float var[kLast]{};

	reco::TrackRef gTrack = muon.globalTrack();
	reco::TrackRef iTrack = muon.innerTrack();
	reco::TrackRef oTrack = muon.outerTrack();

	if(!(muon.innerTrack().isNonnull() and
	     muon.outerTrack().isNonnull() and
	     muon.globalTrack().isNonnull()))
	{
	  return -1;
	}

	//VARIABLE EXTRACTION
	var[kPt] = muon.pt();
	var[kEta] = muon.eta();
	var[kMomID] = -1;
	var[kPID] = -1;

	var[kChi2LocalMomentum] = muon.combinedQuality().chi2LocalMomentum;
	var[kChi2LocalPosition] = muon.combinedQuality().chi2LocalPosition;
	var[kGlbTrackProbability] =  muon.combinedQuality().glbTrackProbability;
	var[kTrkRelChi2] =  muon.combinedQuality().trkRelChi2;

	var[kTrkKink] = muon.combinedQuality().trkKink;
	var[kLog2PlusGlbKink] =  TMath::Log(2+muon.combinedQuality().glbKink);
	var[kSegmentCompatibility] = muon.segmentCompatibility();

	var[kTimeAtIpInOutErr] = muon.time().timeAtIpInOutErr;

	//TRACK RELATED VARIABLES
	
	var[kIValidFraction] =  iTrack->validFraction();
	var[kInnerChi2] =  iTrack->normalizedChi2();
	var[kLayersWithMeasurement] = iTrack->hitPattern().trackerLayersWithMeasurement();

	var[kOuterChi2] = oTrack->normalizedChi2();

	var[kQProd] =  iTrack->charge()*oTrack->charge();

	//vComb Calculation

	const reco::HitPattern &gMpattern = gTrack->hitPattern();

	std::vector<int> fvDThits = {0,0,0,0};
	std::vector<int> fvRPChits = {0,0,0,0};
	std::vector<int> fvCSChits = {0,0,0,0};

	var[kVMuonHitComb] = 0;

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

	  var[kVMuonHitComb] += (fvDThits[station])/2.;
	  var[kVMuonHitComb] += fvRPChits[station];

	  if (fvCSChits[station] > 6){
	    var[kVMuonHitComb] += 6;
	  }else{
	    var[kVMuonHitComb] += fvCSChits[station];
	  }

	}

	if(var[kChi2LocalMomentum] < 5000 and var[kChi2LocalPosition] < 2000 and
	   var[kGlbTrackProbability] < 5000 and var[kTrkKink] < 900 and
	   var[kLog2PlusGlbKink] < 50 and var[kTimeAtIpInOutErr] < 4 and
	   var[kOuterChi2] < 1000 and var[kInnerChi2] < 10 and var[kTrkRelChi2] < 3)
	{
	  return gbrForest_->GetAdaBoostClassifier(var);
	} else {
	  return -1;
        }
}
