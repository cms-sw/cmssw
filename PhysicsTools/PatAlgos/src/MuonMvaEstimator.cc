#include "PhysicsTools/PatAlgos/interface/MuonMvaEstimator.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

using namespace pat;

MuonMvaEstimator::MuonMvaEstimator():
  tmvaReader_("!Color:!Silent:Error"),
  initialized_(false),
  mva_(0),
  dRmax_(0)
{}

void MuonMvaEstimator::initialize(std::string weightsfile,
				  float dRmax)
{ 
  if (initialized_) return;
  tmvaReader_.AddVariable("LepGood_pt",                    &pt_               );
  tmvaReader_.AddVariable("LepGood_eta",                   &eta_              );
  tmvaReader_.AddVariable("LepGood_jetNDauChargedMVASel",  &jetNDauCharged_   );
  tmvaReader_.AddVariable("LepGood_miniRelIsoCharged",     &miniRelIsoCharged_);
  tmvaReader_.AddVariable("LepGood_miniRelIsoNeutral",     &miniRelIsoNeutral_);
  tmvaReader_.AddVariable("LepGood_jetPtRelv2",            &jetPtRel_         );
  tmvaReader_.AddVariable("min(LepGood_jetPtRatiov2,1.5)", &jetPtRatio_       );
  tmvaReader_.AddVariable("max(LepGood_jetBTagCSV,0)",     &jetBTagCSV_       );
  tmvaReader_.AddVariable("LepGood_sip3d",                 &sip_              );
  tmvaReader_.AddVariable("log(abs(LepGood_dxy))",         &log_abs_dxyBS_    ); 
  tmvaReader_.AddVariable("log(abs(LepGood_dz))",          &log_abs_dzPV_     );
  tmvaReader_.AddVariable("LepGood_segmentCompatibility",  &segmentCompatibility_);
  tmvaReader_.BookMVA("BDTG",weightsfile);
  dRmax_ = dRmax;
  initialized_ = true;
};

float ptRel(const reco::Candidate::LorentzVector& muP4, 
	    const reco::Candidate::LorentzVector& jetP4, 
	    bool subtractMuon=true) 
{
  reco::Candidate::LorentzVector jp4 = jetP4;
  if (subtractMuon) jp4-=muP4;
  float dot = muP4.Vect().Dot( jp4.Vect() );
  float ptrel = muP4.P2() - dot*dot/jp4.P2();
  ptrel = ptrel>0 ? sqrt(ptrel) : 0.0;
  return ptrel;
}

void MuonMvaEstimator::computeMva(const pat::Muon& muon,
				  const reco::Vertex& vertex,
				  const reco::JetTagCollection& bTags,
				  const reco::JetCorrector* correctorL1,
				  const reco::JetCorrector* correctorL1L2L3Res)
{
  if (not initialized_) 
    throw cms::Exception("FatalError") << "MuonMVA is not initialized";
  pt_                   = muon.pt();
  eta_                  = muon.eta();
  segmentCompatibility_ = muon.segmentCompatibility();
  miniRelIsoCharged_ = muon.miniPFIsolation().chargedHadronIso();
  miniRelIsoNeutral_ = muon.miniPFIsolation().neutralHadronIso();

  double dB2D  = fabs(muon.dB(pat::Muon::BS2D));
  double dB3D  = muon.dB(pat::Muon::PV3D);
  double edB3D = muon.edB(pat::Muon::PV3D);
  double dz    = fabs(muon.muonBestTrack()->dz(vertex.position()));
  sip_  = edB3D>0?fabs(dB3D/edB3D):0.0; 
  log_abs_dxyBS_     = dB2D>0?log(dB2D):0; 
  log_abs_dzPV_      = dz>0?log(dz):0;

  //Initialise loop variables
  double minDr = 9999;
  double jecL1L2L3Res = 1.;
  double jecL1 = 1.;

  jetPtRatio_ = -99;
  jetPtRel_   = -99;
  jetBTagCSV_ = -999;
  jetNDauCharged_ = -1;

  for (const auto& tagI: bTags){
    // for each muon with the lepton 
    double dr = deltaR(*(tagI.first), muon);
    if(dr > minDr) continue;  
    minDr = dr;
      
    const reco::Candidate::LorentzVector& muP4(muon.p4()); 
    reco::Candidate::LorentzVector jetP4(tagI.first->p4());

    if (correctorL1 && correctorL1L2L3Res){
      jecL1L2L3Res = correctorL1L2L3Res->correction(*(tagI.first));
      jecL1 = correctorL1->correction(*(tagI.first));
    }

    // Get b-jet info
    jetBTagCSV_ = tagI.second;
    jetNDauCharged_ = 0;
    for (auto jet: tagI.first->getJetConstituentsQuick()){
      const reco::PFCandidate *pfcand = dynamic_cast<const reco::PFCandidate*>(jet);
      if (pfcand==nullptr) throw cms::Exception("ConfigurationError") << "Cannot get jet constituents";
      if (pfcand->charge()==0) continue;
      auto bestTrackPtr = pfcand->bestTrack();
      if (!bestTrackPtr) continue;
      if (!bestTrackPtr->quality(reco::Track::highPurity)) continue;
      if (bestTrackPtr->pt()<1.) continue;
      if (bestTrackPtr->hitPattern().numberOfValidHits()<8) continue;
      if (bestTrackPtr->hitPattern().numberOfValidPixelHits()<2) continue;
      if (bestTrackPtr->normalizedChi2()>=5) continue;

      if (std::fabs(bestTrackPtr->dxy(vertex.position())) > 0.2) continue;
      if (std::fabs(bestTrackPtr->dz(vertex.position())) > 17) continue;
      jetNDauCharged_++;
    }

    if(minDr < dRmax_){
      if ((jetP4-muP4).Rho()<0.0001){ 
	jetPtRel_ = 0;
	jetPtRatio_ = 1;
      } else {
	jetP4 -= muP4/jecL1;
	jetP4 *= jecL1L2L3Res;
	jetP4 += muP4;
      
	jetPtRatio_ = muP4.pt()/jetP4.pt();
	jetPtRel_ = ptRel(muP4,jetP4);
      }
    }
  }

  if (jetPtRatio_>1.5) jetPtRatio_ = 1.5;
  if (jetBTagCSV_<0) jetBTagCSV_ = 0;
  mva_ = tmvaReader_.EvaluateMVA("BDTG");
};
