#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"

#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TF1.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include <cmath>

//rounds to first few nonzero sig figs
float rndSF(float value, int nSignificantDigits) 
{
  if(value==0) return 0; 
  
  float dSign = (value > 0.0) ? 1 : -1; 
  value *= dSign; 

  int nOffset = static_cast<int>(log10(value)); 
  if(nOffset>=0) ++nOffset;  

  float dScale = pow(10.0,nSignificantDigits-nOffset); 

  return dSign * static_cast<float>(round(value*dScale))/dScale;    
}

//keeps first n digits after the decimal place
inline float rndDP(float value, int nPlaces)
{
  return float(int(value*pow(10,nPlaces))/pow(10,nPlaces));
}

//----------------------------------------------------------------
//
// DiJet ana Event Data Tree definition
//
class TreePFCandEventData
{
public:
  // ===== Class Methods =====
  void SetDefaults();
  TreePFCandEventData();
  void SetTree(TTree * t) { tree_=t; }
  void SetBranches(int etaBins, int fourierOrder, bool doUEraw = 0);
  void Clear();
  bool doJets;
  bool doMC;

  Float_t         jdphi_;
  // -- particle info --
  Int_t           nPFpart_, nGENpart_, njets_;
  std::vector<Int_t>           pfId_, genPDGId_;
  std::vector<Float_t>         pfEnergy_, jetEnergy_;
  std::vector<Float_t>         pfPt_, genPt_,  jetPt_;
  std::vector<Float_t>         pfEta_, genEta_,  jetEta_;
  std::vector<Float_t>         pfPhi_, genPhi_,  jetPhi_;
  std::vector<Float_t>         pfVsPt_;
  std::vector<Float_t>         pfVsPtInitial_;
  std::vector<Float_t>         pfVsPtEqualized_;
  std::vector<Float_t>         pfArea_;
  Float_t         sumpt[20];
  Float_t         vn[200];
  Float_t         psin[200];
  Float_t         ueraw[1200];

private:
  TTree*                 tree_;
};

class HiPFCandAnalyzer : public edm::EDAnalyzer {
public:
  explicit HiPFCandAnalyzer(const edm::ParameterSet&);
  ~HiPFCandAnalyzer();

  // class methods


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  edm::Service<TFileService> fs;
  edm::Handle<reco::VoronoiMap> backgrounds_;
  edm::Handle<std::vector<float> > vn_;
  edm::Handle<reco::CandidateView> candidates_;

  // === Ana setup ===

  // Event Info
  edm::InputTag pfCandidateLabel_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidatePF_;
  edm::EDGetTokenT<reco::CandidateView> pfCandidateView_;
  edm::EDGetTokenT<reco::GenParticleCollection> genLabel_;
  edm::EDGetTokenT<pat::JetCollection> jetLabel_;
  edm::EDGetTokenT<std::vector<float> > srcVorFloat_;
  edm::EDGetTokenT<reco::VoronoiMap> srcVorMap_;

  TTree	  *pfTree_;
  TreePFCandEventData pfEvt_;

  // cuts
  Double_t        pfPtMin_;
  Double_t        jetPtMin_;
  Double_t        genPtMin_;

  int           fourierOrder_;
  int           etaBins_;

  // debug
  Int_t	  verbosity_;

  bool   doJets_;
  bool   doMC_;
  bool   doVS_;
  bool   doUEraw_;
  bool   skipCharged_;
};
