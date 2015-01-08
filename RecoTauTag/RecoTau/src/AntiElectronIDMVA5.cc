#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA5.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <TFile.h>

AntiElectronIDMVA5::AntiElectronIDMVA5(const edm::ParameterSet& cfg)
  : isInitialized_(false),
    mva_NoEleMatch_woGwoGSF_BL_(0),
    mva_NoEleMatch_woGwGSF_BL_(0),
    mva_NoEleMatch_wGwoGSF_BL_(0),
    mva_NoEleMatch_wGwGSF_BL_(0),
    mva_woGwoGSF_BL_(0),
    mva_woGwGSF_BL_(0),
    mva_wGwoGSF_BL_(0),
    mva_wGwGSF_BL_(0),
    mva_NoEleMatch_woGwoGSF_EC_(0),
    mva_NoEleMatch_woGwGSF_EC_(0),
    mva_NoEleMatch_wGwoGSF_EC_(0),
    mva_NoEleMatch_wGwGSF_EC_(0),
    mva_woGwoGSF_EC_(0),
    mva_woGwGSF_EC_(0),
    mva_wGwoGSF_EC_(0),
    mva_wGwGSF_EC_(0)
{
  loadMVAfromDB_ = cfg.exists("loadMVAfromDB") ? cfg.getParameter<bool>("loadMVAfromDB"): false;
  if ( !loadMVAfromDB_ ) {
    if(cfg.exists("inputFileName")){
      inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
    }else throw cms::Exception("MVA input not defined") << "Requested to load tau MVA input from ROOT file but no file provided in cfg file";
    
  }
  mvaName_NoEleMatch_woGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwoGSF_BL");
  mvaName_NoEleMatch_woGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwGSF_BL");
  mvaName_NoEleMatch_wGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwoGSF_BL");
  mvaName_NoEleMatch_wGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwGSF_BL");
  mvaName_woGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_woGwoGSF_BL");
  mvaName_woGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_woGwGSF_BL");
  mvaName_wGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_wGwoGSF_BL");
  mvaName_wGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_wGwGSF_BL");
  mvaName_NoEleMatch_woGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwoGSF_EC");
  mvaName_NoEleMatch_woGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwGSF_EC");
  mvaName_NoEleMatch_wGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwoGSF_EC");
  mvaName_NoEleMatch_wGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwGSF_EC");
  mvaName_woGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_woGwoGSF_EC");
  mvaName_woGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_woGwGSF_EC");
  mvaName_wGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_wGwoGSF_EC");
  mvaName_wGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_wGwGSF_EC");

  Var_NoEleMatch_woGwoGSF_Barrel_ = new Float_t[10];
  Var_NoEleMatch_woGwGSF_Barrel_ = new Float_t[16];
  Var_NoEleMatch_wGwoGSF_Barrel_ = new Float_t[14];
  Var_NoEleMatch_wGwGSF_Barrel_ = new Float_t[20];
  Var_woGwoGSF_Barrel_ = new Float_t[18];
  Var_woGwGSF_Barrel_ = new Float_t[24];
  Var_wGwoGSF_Barrel_ = new Float_t[22];
  Var_wGwGSF_Barrel_ = new Float_t[28];
  Var_NoEleMatch_woGwoGSF_Endcap_ = new Float_t[9];
  Var_NoEleMatch_woGwGSF_Endcap_ = new Float_t[15];
  Var_NoEleMatch_wGwoGSF_Endcap_ = new Float_t[13];
  Var_NoEleMatch_wGwGSF_Endcap_ = new Float_t[19];
  Var_woGwoGSF_Endcap_ = new Float_t[17];
  Var_woGwGSF_Endcap_ = new Float_t[23];
  Var_wGwoGSF_Endcap_ = new Float_t[21];
  Var_wGwGSF_Endcap_ = new Float_t[27];
    
  verbosity_ = 0;
}

AntiElectronIDMVA5::~AntiElectronIDMVA5()
{
  delete [] Var_NoEleMatch_woGwoGSF_Barrel_;
  delete [] Var_NoEleMatch_woGwGSF_Barrel_;
  delete [] Var_NoEleMatch_wGwoGSF_Barrel_;
  delete [] Var_NoEleMatch_wGwGSF_Barrel_;
  delete [] Var_woGwoGSF_Barrel_;
  delete [] Var_woGwGSF_Barrel_;
  delete [] Var_wGwoGSF_Barrel_;
  delete [] Var_wGwGSF_Barrel_;
  delete [] Var_NoEleMatch_woGwoGSF_Endcap_;
  delete [] Var_NoEleMatch_woGwGSF_Endcap_;
  delete [] Var_NoEleMatch_wGwoGSF_Endcap_;
  delete [] Var_NoEleMatch_wGwGSF_Endcap_;
  delete [] Var_woGwoGSF_Endcap_;
  delete [] Var_woGwGSF_Endcap_;
  delete [] Var_wGwoGSF_Endcap_;
  delete [] Var_wGwGSF_Endcap_;
  if ( !loadMVAfromDB_ ){
    delete mva_NoEleMatch_woGwoGSF_BL_;
    delete mva_NoEleMatch_woGwGSF_BL_;
    delete mva_NoEleMatch_wGwoGSF_BL_;
    delete mva_NoEleMatch_wGwGSF_BL_;
    delete mva_woGwoGSF_BL_;
    delete mva_woGwGSF_BL_;
    delete mva_wGwoGSF_BL_;
    delete mva_wGwGSF_BL_;
    delete mva_NoEleMatch_woGwoGSF_EC_;
    delete mva_NoEleMatch_woGwGSF_EC_;
    delete mva_NoEleMatch_wGwoGSF_EC_;
    delete mva_NoEleMatch_wGwGSF_EC_;
    delete mva_woGwoGSF_EC_;
    delete mva_woGwGSF_EC_;
    delete mva_wGwoGSF_EC_;
    delete mva_wGwGSF_EC_;
  }
  for ( std::vector<TFile*>::iterator it = inputFilesToDelete_.begin();
	it != inputFilesToDelete_.end(); ++it ) {
    delete (*it);
  }
}

namespace
{
  const GBRForest* loadMVAfromFile(TFile* inputFile, const std::string& mvaName)
  {
  
    //const GBRForest* mva = dynamic_cast<GBRForest*>(inputFile->Get(mvaName.data())); // CV: dynamic_cast<GBRForest*> fails for some reason ?!
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if ( !mva )
      throw cms::Exception("PFRecoTauDiscriminationAgainstMuonMVA::loadMVA")
        << " Failed to load MVA = " << mvaName.data() << " from file " << " !!\n";

    return mva;
  }

  const GBRForest* loadMVAfromDB(const edm::EventSetup& es, const std::string& mvaName)
  {
    edm::ESHandle<GBRForest> mva;
    es.get<GBRWrapperRcd>().get(mvaName, mva);
    return mva.product();
  }
}

void AntiElectronIDMVA5::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( !isInitialized_ ) {
    if ( loadMVAfromDB_ ) {
      mva_NoEleMatch_woGwoGSF_BL_ = loadMVAfromDB(es, mvaName_NoEleMatch_woGwoGSF_BL_);
      mva_NoEleMatch_woGwGSF_BL_  = loadMVAfromDB(es, mvaName_NoEleMatch_woGwGSF_BL_);
      mva_NoEleMatch_wGwoGSF_BL_  = loadMVAfromDB(es, mvaName_NoEleMatch_wGwoGSF_BL_);
      mva_NoEleMatch_wGwGSF_BL_   = loadMVAfromDB(es, mvaName_NoEleMatch_wGwGSF_BL_);
      mva_woGwoGSF_BL_            = loadMVAfromDB(es, mvaName_woGwoGSF_BL_);
      mva_woGwGSF_BL_             = loadMVAfromDB(es, mvaName_woGwGSF_BL_);
      mva_wGwoGSF_BL_             = loadMVAfromDB(es, mvaName_wGwoGSF_BL_);
      mva_wGwGSF_BL_              = loadMVAfromDB(es, mvaName_wGwGSF_BL_);
      mva_NoEleMatch_woGwoGSF_EC_ = loadMVAfromDB(es, mvaName_NoEleMatch_woGwoGSF_EC_);
      mva_NoEleMatch_woGwGSF_EC_  = loadMVAfromDB(es, mvaName_NoEleMatch_woGwGSF_EC_);
      mva_NoEleMatch_wGwoGSF_EC_  = loadMVAfromDB(es, mvaName_NoEleMatch_wGwoGSF_EC_);
      mva_NoEleMatch_wGwGSF_EC_   = loadMVAfromDB(es, mvaName_NoEleMatch_wGwGSF_EC_);
      mva_woGwoGSF_EC_            = loadMVAfromDB(es, mvaName_woGwoGSF_EC_);
      mva_woGwGSF_EC_             = loadMVAfromDB(es, mvaName_woGwGSF_EC_);
      mva_wGwoGSF_EC_             = loadMVAfromDB(es, mvaName_wGwoGSF_EC_);
      mva_wGwGSF_EC_              = loadMVAfromDB(es, mvaName_wGwGSF_EC_);  
    } else {
          if ( inputFileName_.location() == edm::FileInPath::Unknown ) throw cms::Exception("PFRecoTauDiscriminationAgainstMuonMVA::loadMVA")
          << " Failed to find File = " << inputFileName_ << " !!\n";
          TFile* inputFile = new TFile(inputFileName_.fullPath().data());

      mva_NoEleMatch_woGwoGSF_BL_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwoGSF_BL_);
      mva_NoEleMatch_woGwGSF_BL_  = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwGSF_BL_);
      mva_NoEleMatch_wGwoGSF_BL_  = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwoGSF_BL_);
      mva_NoEleMatch_wGwGSF_BL_   = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwGSF_BL_);
      mva_woGwoGSF_BL_            = loadMVAfromFile(inputFile, mvaName_woGwoGSF_BL_);
      mva_woGwGSF_BL_             = loadMVAfromFile(inputFile, mvaName_woGwGSF_BL_);
      mva_wGwoGSF_BL_             = loadMVAfromFile(inputFile, mvaName_wGwoGSF_BL_);
      mva_wGwGSF_BL_              = loadMVAfromFile(inputFile, mvaName_wGwGSF_BL_);
      mva_NoEleMatch_woGwoGSF_EC_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwoGSF_EC_);
      mva_NoEleMatch_woGwGSF_EC_  = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwGSF_EC_);
      mva_NoEleMatch_wGwoGSF_EC_  = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwoGSF_EC_);
      mva_NoEleMatch_wGwGSF_EC_   = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwGSF_EC_);
      mva_woGwoGSF_EC_            = loadMVAfromFile(inputFile, mvaName_woGwoGSF_EC_);
      mva_woGwGSF_EC_             = loadMVAfromFile(inputFile, mvaName_woGwGSF_EC_);
      mva_wGwoGSF_EC_             = loadMVAfromFile(inputFile, mvaName_wGwoGSF_EC_);
      mva_wGwGSF_EC_              = loadMVAfromFile(inputFile, mvaName_wGwGSF_EC_);
      inputFilesToDelete_.push_back(inputFile);  
    }
    isInitialized_ = true;
  }
}

double AntiElectronIDMVA5::MVAValue(Float_t TauEtaAtEcalEntrance,
				    Float_t TauPt,
				    Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
				    Float_t TauLeadChargedPFCandPt,
				    Float_t TaudCrackEta,
				    Float_t TaudCrackPhi,
				    Float_t TauEmFraction,
				    Float_t TauSignalPFGammaCands,
				    Float_t TauLeadPFChargedHadrHoP,
				    Float_t TauLeadPFChargedHadrEoP,
				    Float_t TauVisMass,
				    Float_t TauHadrMva,
				    const std::vector<Float_t>& GammasdEta,
				    const std::vector<Float_t>& GammasdPhi,
				    const std::vector<Float_t>& GammasPt,
				    Float_t TauKFNumHits,				   
				    Float_t TauGSFNumHits,				   
				    Float_t TauGSFChi2,				   
				    Float_t TauGSFTrackResol,
				    Float_t TauGSFTracklnPt,
				    Float_t TauGSFTrackEta,
				    Float_t TauPhi,
				    Float_t TauSignalPFChargedCands,
				    Float_t TauHasGsf,
				    Float_t ElecEta,
				    Float_t ElecPhi,
				    Float_t ElecPt,
				    Float_t ElecEe,
				    Float_t ElecEgamma,
				    Float_t ElecPin,
				    Float_t ElecPout,
				    Float_t ElecFbrem,
				    Float_t ElecChi2GSF,
				    Float_t ElecGSFNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta)
{
  double sumPt  = 0.;
  double dEta   = 0.;
  double dEta2  = 0.;
  double dPhi   = 0.;
  double dPhi2  = 0.;
  double sumPt2 = 0.;
  for ( unsigned int i = 0 ; i < GammasPt.size() ; ++i ) {
    double pt_i  = GammasPt[i];
    double phi_i = GammasdPhi[i];
    if ( GammasdPhi[i] > M_PI ) phi_i = GammasdPhi[i] - 2*M_PI;
    else if ( GammasdPhi[i] < -M_PI ) phi_i = GammasdPhi[i] + 2*M_PI;
    double eta_i = GammasdEta[i];
    sumPt  +=  pt_i;
    sumPt2 += (pt_i*pt_i);
    dEta   += (pt_i*eta_i);
    dEta2  += (pt_i*eta_i*eta_i);
    dPhi   += (pt_i*phi_i);
    dPhi2  += (pt_i*phi_i*phi_i);
  }

  Float_t TauGammaEnFrac = sumPt/TauPt;

  if ( sumPt > 0. ) {
    dEta  /= sumPt;
    dPhi  /= sumPt;
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }

  Float_t TauGammaEtaMom = std::sqrt(dEta2)*std::sqrt(TauGammaEnFrac)*TauPt;
  Float_t TauGammaPhiMom = std::sqrt(dPhi2)*std::sqrt(TauGammaEnFrac)*TauPt;

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPt,
		  TauLeadChargedPFCandEtaAtEcalEntrance,
		  TauLeadChargedPFCandPt,
		  TaudCrackEta,
		  TaudCrackPhi,
		  TauEmFraction,
		  TauSignalPFGammaCands,				    
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauVisMass,
		  TauHadrMva,
		  TauGammaEtaMom,
		  TauGammaPhiMom,
		  TauGammaEnFrac,
		  TauKFNumHits,				   
		  TauGSFNumHits,				   
		  TauGSFChi2,				   
		  TauGSFTrackResol,
		  TauGSFTracklnPt,
		  TauGSFTrackEta,
		  TauPhi,
		  TauSignalPFChargedCands,
		  TauHasGsf,
		  ElecEta,
		  ElecPhi,
		  ElecPt,
		  ElecEe,
		  ElecEgamma,
		  ElecPin,
		  ElecPout,
		  ElecFbrem,
		  ElecChi2GSF,
		  ElecGSFNumHits,
		  ElecGSFTrackResol,
		  ElecGSFTracklnPt,
		  ElecGSFTrackEta);
}

double AntiElectronIDMVA5::MVAValue(Float_t TauEtaAtEcalEntrance,
				    Float_t TauPt,
				    Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
				    Float_t TauLeadChargedPFCandPt,
				    Float_t TaudCrackEta,
				    Float_t TaudCrackPhi,
				    Float_t TauEmFract,
				    Float_t TauSignalPFGammaCands,				    
				    Float_t TauLeadPFChargedHadrHoP,
				    Float_t TauLeadPFChargedHadrEoP,
				    Float_t TauVisMass,
				    Float_t TauHadrMva,
				    Float_t TauGammaEtaMom,
				    Float_t TauGammaPhiMom,
				    Float_t TauGammaEnFrac,
				    Float_t TauKFNumHits,				   
				    Float_t TauGSFNumHits,				   
				    Float_t TauGSFChi2,				   
				    Float_t TauGSFTrackResol,
				    Float_t TauGSFTracklnPt,
				    Float_t TauGSFTrackEta,
				    Float_t TauPhi,
				    Float_t TauSignalPFChargedCands,
				    Float_t TauHasGsf,
				    Float_t ElecEta,
				    Float_t ElecPhi,
				    Float_t ElecPt,
				    Float_t ElecEe,
				    Float_t ElecEgamma,
				    Float_t ElecPin,
				    Float_t ElecPout,
				    Float_t ElecFbrem,
				    Float_t ElecChi2GSF,
				    Float_t ElecGSFNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta)
{
  if ( !isInitialized_ ) {
    throw cms::Exception("ClassNotInitialized")
      << " AntiElectronMVA not properly initialized !!\n";
  }

  Float_t TauEmFraction = std::max(TauEmFract, float(0.));
  Float_t TauNumHitsVariable = (TauGSFNumHits - TauKFNumHits)/(TauGSFNumHits + TauKFNumHits); 
  Float_t ElecEtotOverPin = (ElecEe + ElecEgamma)/ElecPin;
  Float_t ElecEgammaOverPdif = ElecEgamma/(ElecPin - ElecPout);

  double mvaValue = -99.;
  if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 && TauSignalPFGammaCands == 0 && TauHasGsf < 0.5) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ){
      Var_NoEleMatch_woGwoGSF_Barrel_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[1] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[2] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwoGSF_Barrel_[3] = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwoGSF_Barrel_[4] = TauEmFraction;
      Var_NoEleMatch_woGwoGSF_Barrel_[5] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[6] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[7] = TauVisMass;
      Var_NoEleMatch_woGwoGSF_Barrel_[8] = TaudCrackEta;
      Var_NoEleMatch_woGwoGSF_Barrel_[9] = TaudCrackPhi;
      mvaValue = mva_NoEleMatch_woGwoGSF_BL_->GetClassifier(Var_NoEleMatch_woGwoGSF_Barrel_);
    } else {
      Var_NoEleMatch_woGwoGSF_Endcap_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Endcap_[1] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Endcap_[2] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwoGSF_Endcap_[3] = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwoGSF_Endcap_[4] = TauEmFraction;
      Var_NoEleMatch_woGwoGSF_Endcap_[5] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Endcap_[6] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Endcap_[7] = TauVisMass;
      Var_NoEleMatch_woGwoGSF_Endcap_[8] = TaudCrackEta;
      mvaValue = mva_NoEleMatch_woGwoGSF_EC_->GetClassifier(Var_NoEleMatch_woGwoGSF_Endcap_);
    }
  } else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 && TauSignalPFGammaCands == 0 && TauHasGsf > 0.5) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ){
      Var_NoEleMatch_woGwGSF_Barrel_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwGSF_Barrel_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwGSF_Barrel_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwGSF_Barrel_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwGSF_Barrel_[4]  = TauEmFraction;
      Var_NoEleMatch_woGwGSF_Barrel_[5]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwGSF_Barrel_[6]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwGSF_Barrel_[7]  = TauVisMass;
      Var_NoEleMatch_woGwGSF_Barrel_[8]  = TauHadrMva;
      Var_NoEleMatch_woGwGSF_Barrel_[9]  = TauGSFChi2;
      Var_NoEleMatch_woGwGSF_Barrel_[10] = TauNumHitsVariable;
      Var_NoEleMatch_woGwGSF_Barrel_[11] = TauGSFTrackResol;
      Var_NoEleMatch_woGwGSF_Barrel_[12] = TauGSFTracklnPt;
      Var_NoEleMatch_woGwGSF_Barrel_[13] = TauGSFTrackEta;
      Var_NoEleMatch_woGwGSF_Barrel_[14] = TaudCrackEta;
      Var_NoEleMatch_woGwGSF_Barrel_[15] = TaudCrackPhi;
      mvaValue = mva_NoEleMatch_woGwGSF_BL_->GetClassifier(Var_NoEleMatch_woGwGSF_Barrel_);
    } else {
      Var_NoEleMatch_woGwGSF_Endcap_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwGSF_Endcap_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwGSF_Endcap_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwGSF_Endcap_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwGSF_Endcap_[4]  = TauEmFraction;
      Var_NoEleMatch_woGwGSF_Endcap_[5]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwGSF_Endcap_[6]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwGSF_Endcap_[7]  = TauVisMass;
      Var_NoEleMatch_woGwGSF_Endcap_[8]  = TauHadrMva;
      Var_NoEleMatch_woGwGSF_Endcap_[9]  = TauGSFChi2;
      Var_NoEleMatch_woGwGSF_Endcap_[10] = TauNumHitsVariable;
      Var_NoEleMatch_woGwGSF_Endcap_[11] = TauGSFTrackResol;
      Var_NoEleMatch_woGwGSF_Endcap_[12] = TauGSFTracklnPt;
      Var_NoEleMatch_woGwGSF_Endcap_[13] = TauGSFTrackEta;
      Var_NoEleMatch_woGwGSF_Endcap_[14] = TaudCrackEta;
      mvaValue = mva_NoEleMatch_woGwGSF_EC_->GetClassifier(Var_NoEleMatch_woGwGSF_Endcap_);
    }
  } else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 && TauSignalPFGammaCands > 0 && TauHasGsf < 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ){
      Var_NoEleMatch_wGwoGSF_Barrel_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Barrel_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Barrel_[4]  = TauEmFraction;
      Var_NoEleMatch_wGwoGSF_Barrel_[5]  = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwoGSF_Barrel_[6]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[7]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[8]  = TauVisMass;
      Var_NoEleMatch_wGwoGSF_Barrel_[9]  = TauGammaEtaMom;
      Var_NoEleMatch_wGwoGSF_Barrel_[10] = TauGammaPhiMom;
      Var_NoEleMatch_wGwoGSF_Barrel_[11] = TauGammaEnFrac;
      Var_NoEleMatch_wGwoGSF_Barrel_[12] = TaudCrackEta;
      Var_NoEleMatch_wGwoGSF_Barrel_[13] = TaudCrackPhi;
      mvaValue = mva_NoEleMatch_wGwoGSF_BL_->GetClassifier(Var_NoEleMatch_wGwoGSF_Barrel_);
    } else {
      Var_NoEleMatch_wGwoGSF_Endcap_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Endcap_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Endcap_[4]  = TauEmFraction;
      Var_NoEleMatch_wGwoGSF_Endcap_[5]  = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwoGSF_Endcap_[6]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[7]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[8]  = TauVisMass;
      Var_NoEleMatch_wGwoGSF_Endcap_[9]  = TauGammaEtaMom;
      Var_NoEleMatch_wGwoGSF_Endcap_[10] = TauGammaPhiMom;
      Var_NoEleMatch_wGwoGSF_Endcap_[11] = TauGammaEnFrac;
      Var_NoEleMatch_wGwoGSF_Endcap_[12] = TaudCrackEta;
      mvaValue = mva_NoEleMatch_wGwoGSF_EC_->GetClassifier(Var_NoEleMatch_wGwoGSF_Endcap_);
    }
  } 
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 && TauSignalPFGammaCands > 0 && TauHasGsf > 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ) {
      Var_NoEleMatch_wGwGSF_Barrel_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwGSF_Barrel_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwGSF_Barrel_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwGSF_Barrel_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwGSF_Barrel_[4]  = TauEmFraction;
      Var_NoEleMatch_wGwGSF_Barrel_[5]  = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwGSF_Barrel_[6]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwGSF_Barrel_[7]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwGSF_Barrel_[8]  = TauVisMass;
      Var_NoEleMatch_wGwGSF_Barrel_[9]  = TauHadrMva;
      Var_NoEleMatch_wGwGSF_Barrel_[10] = TauGammaEtaMom;
      Var_NoEleMatch_wGwGSF_Barrel_[11] = TauGammaPhiMom;
      Var_NoEleMatch_wGwGSF_Barrel_[12] = TauGammaEnFrac;
      Var_NoEleMatch_wGwGSF_Barrel_[13] = TauGSFChi2;
      Var_NoEleMatch_wGwGSF_Barrel_[14] = TauNumHitsVariable;
      Var_NoEleMatch_wGwGSF_Barrel_[15] = TauGSFTrackResol;
      Var_NoEleMatch_wGwGSF_Barrel_[16] = TauGSFTracklnPt;
      Var_NoEleMatch_wGwGSF_Barrel_[17] = TauGSFTrackEta;
      Var_NoEleMatch_wGwGSF_Barrel_[18] = TaudCrackEta;
      Var_NoEleMatch_wGwGSF_Barrel_[19] = TaudCrackPhi;
      mvaValue =  mva_NoEleMatch_wGwGSF_BL_->GetClassifier(Var_NoEleMatch_wGwGSF_Barrel_);
    } else {
      Var_NoEleMatch_wGwGSF_Endcap_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwGSF_Endcap_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwGSF_Endcap_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwGSF_Endcap_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwGSF_Endcap_[4]  = TauEmFraction;
      Var_NoEleMatch_wGwGSF_Endcap_[5]  = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwGSF_Endcap_[6]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwGSF_Endcap_[7]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwGSF_Endcap_[8]  = TauVisMass;
      Var_NoEleMatch_wGwGSF_Endcap_[9]  = TauHadrMva;
      Var_NoEleMatch_wGwGSF_Endcap_[10] = TauGammaEtaMom;
      Var_NoEleMatch_wGwGSF_Endcap_[11] = TauGammaPhiMom;
      Var_NoEleMatch_wGwGSF_Endcap_[12] = TauGammaEnFrac;
      Var_NoEleMatch_wGwGSF_Endcap_[13] = TauGSFChi2;
      Var_NoEleMatch_wGwGSF_Endcap_[14] = TauNumHitsVariable;
      Var_NoEleMatch_wGwGSF_Endcap_[15] = TauGSFTrackResol;
      Var_NoEleMatch_wGwGSF_Endcap_[16] = TauGSFTracklnPt;
      Var_NoEleMatch_wGwGSF_Endcap_[17] = TauGSFTrackEta;
      Var_NoEleMatch_wGwGSF_Endcap_[18] = TaudCrackEta;
      mvaValue = mva_NoEleMatch_wGwGSF_EC_->GetClassifier(Var_NoEleMatch_wGwGSF_Endcap_);
    } 
  } else if ( TauSignalPFGammaCands == 0 && TauHasGsf < 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ) {
      Var_woGwoGSF_Barrel_[0]  = ElecEtotOverPin;
      Var_woGwoGSF_Barrel_[1]  = ElecEgammaOverPdif;
      Var_woGwoGSF_Barrel_[2]  = ElecFbrem;
      Var_woGwoGSF_Barrel_[3]  = ElecChi2GSF;
      Var_woGwoGSF_Barrel_[4]  = ElecGSFNumHits;
      Var_woGwoGSF_Barrel_[5]  = ElecGSFTrackResol;
      Var_woGwoGSF_Barrel_[6]  = ElecGSFTracklnPt;
      Var_woGwoGSF_Barrel_[7]  = ElecGSFTrackEta;
      Var_woGwoGSF_Barrel_[8]  = TauEtaAtEcalEntrance;
      Var_woGwoGSF_Barrel_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_woGwoGSF_Barrel_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_woGwoGSF_Barrel_[11] = std::log(std::max(float(1.), TauPt));
      Var_woGwoGSF_Barrel_[12] = TauEmFraction;
      Var_woGwoGSF_Barrel_[13] = TauLeadPFChargedHadrHoP;
      Var_woGwoGSF_Barrel_[14] = TauLeadPFChargedHadrEoP;
      Var_woGwoGSF_Barrel_[15] = TauVisMass;
      Var_woGwoGSF_Barrel_[16] = TaudCrackEta;
      Var_woGwoGSF_Barrel_[17] = TaudCrackPhi;
      mvaValue = mva_woGwoGSF_BL_->GetClassifier(Var_woGwoGSF_Barrel_);
    } else {
      Var_woGwoGSF_Endcap_[0]  = ElecEtotOverPin;
      Var_woGwoGSF_Endcap_[1]  = ElecEgammaOverPdif;
      Var_woGwoGSF_Endcap_[2]  = ElecFbrem;
      Var_woGwoGSF_Endcap_[3]  = ElecChi2GSF;
      Var_woGwoGSF_Endcap_[4]  = ElecGSFNumHits;
      Var_woGwoGSF_Endcap_[5]  = ElecGSFTrackResol;
      Var_woGwoGSF_Endcap_[6]  = ElecGSFTracklnPt;
      Var_woGwoGSF_Endcap_[7]  = ElecGSFTrackEta;
      Var_woGwoGSF_Endcap_[8]  = TauEtaAtEcalEntrance;
      Var_woGwoGSF_Endcap_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_woGwoGSF_Endcap_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_woGwoGSF_Endcap_[11] = std::log(std::max(float(1.), TauPt));
      Var_woGwoGSF_Endcap_[12] = TauEmFraction;
      Var_woGwoGSF_Endcap_[13] = TauLeadPFChargedHadrHoP;
      Var_woGwoGSF_Endcap_[14] = TauLeadPFChargedHadrEoP;
      Var_woGwoGSF_Endcap_[15] = TauVisMass;
      Var_woGwoGSF_Endcap_[16] = TaudCrackEta;
      mvaValue = mva_woGwoGSF_EC_->GetClassifier(Var_woGwoGSF_Endcap_);
    }
  } else if ( TauSignalPFGammaCands == 0 && TauHasGsf > 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ) {
      Var_woGwGSF_Barrel_[0]  = ElecEtotOverPin;
      Var_woGwGSF_Barrel_[1]  = ElecEgammaOverPdif;
      Var_woGwGSF_Barrel_[2]  = ElecFbrem;
      Var_woGwGSF_Barrel_[3]  = ElecChi2GSF;
      Var_woGwGSF_Barrel_[4]  = ElecGSFNumHits;
      Var_woGwGSF_Barrel_[5]  = ElecGSFTrackResol;
      Var_woGwGSF_Barrel_[6]  = ElecGSFTracklnPt;
      Var_woGwGSF_Barrel_[7]  = ElecGSFTrackEta;
      Var_woGwGSF_Barrel_[8]  = TauEtaAtEcalEntrance;
      Var_woGwGSF_Barrel_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_woGwGSF_Barrel_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_woGwGSF_Barrel_[11] = std::log(std::max(float(1.), TauPt));
      Var_woGwGSF_Barrel_[12] = TauEmFraction;
      Var_woGwGSF_Barrel_[13] = TauLeadPFChargedHadrHoP;
      Var_woGwGSF_Barrel_[14] = TauLeadPFChargedHadrEoP;
      Var_woGwGSF_Barrel_[15] = TauVisMass;
      Var_woGwGSF_Barrel_[16] = TauHadrMva;
      Var_woGwGSF_Barrel_[17] = TauGSFChi2;
      Var_woGwGSF_Barrel_[18] = TauNumHitsVariable;
      Var_woGwGSF_Barrel_[19] = TauGSFTrackResol;
      Var_woGwGSF_Barrel_[20] = TauGSFTracklnPt;
      Var_woGwGSF_Barrel_[21] = TauGSFTrackEta;
      Var_woGwGSF_Barrel_[22] = TaudCrackEta;
      Var_woGwGSF_Barrel_[23] = TaudCrackPhi;
      mvaValue = mva_woGwGSF_BL_->GetClassifier(Var_woGwGSF_Barrel_);
    } else {
      Var_woGwGSF_Endcap_[0]  = ElecEtotOverPin;
      Var_woGwGSF_Endcap_[1]  = ElecEgammaOverPdif;
      Var_woGwGSF_Endcap_[2]  = ElecFbrem;
      Var_woGwGSF_Endcap_[3]  = ElecChi2GSF;
      Var_woGwGSF_Endcap_[4]  = ElecGSFNumHits;
      Var_woGwGSF_Endcap_[5]  = ElecGSFTrackResol;
      Var_woGwGSF_Endcap_[6]  = ElecGSFTracklnPt;
      Var_woGwGSF_Endcap_[7]  = ElecGSFTrackEta;
      Var_woGwGSF_Endcap_[8]  = TauEtaAtEcalEntrance;
      Var_woGwGSF_Endcap_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_woGwGSF_Endcap_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_woGwGSF_Endcap_[11] = std::log(std::max(float(1.), TauPt));
      Var_woGwGSF_Endcap_[12] = TauEmFraction;
      Var_woGwGSF_Endcap_[13] = TauLeadPFChargedHadrHoP;
      Var_woGwGSF_Endcap_[14] = TauLeadPFChargedHadrEoP;
      Var_woGwGSF_Endcap_[15] = TauVisMass;
      Var_woGwGSF_Endcap_[16] = TauHadrMva;
      Var_woGwGSF_Endcap_[17] = TauGSFChi2;
      Var_woGwGSF_Endcap_[18] = TauNumHitsVariable;
      Var_woGwGSF_Endcap_[19] = TauGSFTrackResol;
      Var_woGwGSF_Endcap_[20] = TauGSFTracklnPt;
      Var_woGwGSF_Endcap_[21] = TauGSFTrackEta;
      Var_woGwGSF_Endcap_[22] = TaudCrackEta;
      mvaValue = mva_woGwGSF_EC_->GetClassifier(Var_woGwGSF_Endcap_);
    } 
  } else if ( TauSignalPFGammaCands > 0 && TauHasGsf < 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ) {
      Var_wGwoGSF_Barrel_[0]  = ElecEtotOverPin;
      Var_wGwoGSF_Barrel_[1]  = ElecEgammaOverPdif;
      Var_wGwoGSF_Barrel_[2]  = ElecFbrem;
      Var_wGwoGSF_Barrel_[3]  = ElecChi2GSF;
      Var_wGwoGSF_Barrel_[4]  = ElecGSFNumHits;
      Var_wGwoGSF_Barrel_[5]  = ElecGSFTrackResol;
      Var_wGwoGSF_Barrel_[6]  = ElecGSFTracklnPt;
      Var_wGwoGSF_Barrel_[7]  = ElecGSFTrackEta;
      Var_wGwoGSF_Barrel_[8]  = TauEtaAtEcalEntrance;
      Var_wGwoGSF_Barrel_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_wGwoGSF_Barrel_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_wGwoGSF_Barrel_[11] = std::log(std::max(float(1.), TauPt));
      Var_wGwoGSF_Barrel_[12] = TauEmFraction;
      Var_wGwoGSF_Barrel_[13] = TauSignalPFGammaCands;
      Var_wGwoGSF_Barrel_[14] = TauLeadPFChargedHadrHoP;
      Var_wGwoGSF_Barrel_[15] = TauLeadPFChargedHadrEoP;
      Var_wGwoGSF_Barrel_[16] = TauVisMass;
      Var_wGwoGSF_Barrel_[17] = TauGammaEtaMom;
      Var_wGwoGSF_Barrel_[18] = TauGammaPhiMom;
      Var_wGwoGSF_Barrel_[19] = TauGammaEnFrac;
      Var_wGwoGSF_Barrel_[20] = TaudCrackEta;
      Var_wGwoGSF_Barrel_[21] = TaudCrackPhi;
      mvaValue = mva_wGwoGSF_BL_->GetClassifier(Var_wGwoGSF_Barrel_);
    } else {
      Var_wGwoGSF_Endcap_[0]  = ElecEtotOverPin;
      Var_wGwoGSF_Endcap_[1]  = ElecEgammaOverPdif;
      Var_wGwoGSF_Endcap_[2]  = ElecFbrem;
      Var_wGwoGSF_Endcap_[3]  = ElecChi2GSF;
      Var_wGwoGSF_Endcap_[4]  = ElecGSFNumHits;
      Var_wGwoGSF_Endcap_[5]  = ElecGSFTrackResol;
      Var_wGwoGSF_Endcap_[6]  = ElecGSFTracklnPt;
      Var_wGwoGSF_Endcap_[7]  = ElecGSFTrackEta;
      Var_wGwoGSF_Endcap_[8]  = TauEtaAtEcalEntrance;
      Var_wGwoGSF_Endcap_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_wGwoGSF_Endcap_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_wGwoGSF_Endcap_[11] = std::log(std::max(float(1.), TauPt));
      Var_wGwoGSF_Endcap_[12] = TauEmFraction;
      Var_wGwoGSF_Endcap_[13] = TauSignalPFGammaCands;
      Var_wGwoGSF_Endcap_[14] = TauLeadPFChargedHadrHoP;
      Var_wGwoGSF_Endcap_[15] = TauLeadPFChargedHadrEoP;
      Var_wGwoGSF_Endcap_[16] = TauVisMass;
      Var_wGwoGSF_Endcap_[17] = TauGammaEtaMom;
      Var_wGwoGSF_Endcap_[18] = TauGammaPhiMom;
      Var_wGwoGSF_Endcap_[19] = TauGammaEnFrac;
      Var_wGwoGSF_Endcap_[20] = TaudCrackEta;
      mvaValue = mva_wGwoGSF_EC_->GetClassifier(Var_wGwoGSF_Endcap_);
    }
  } else if ( TauSignalPFGammaCands > 0 && TauHasGsf > 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < 1.479 ) {
      Var_wGwGSF_Barrel_[0]  = ElecEtotOverPin;
      Var_wGwGSF_Barrel_[1]  = ElecEgammaOverPdif;
      Var_wGwGSF_Barrel_[2]  = ElecFbrem;
      Var_wGwGSF_Barrel_[3]  = ElecChi2GSF;
      Var_wGwGSF_Barrel_[4]  = ElecGSFNumHits;
      Var_wGwGSF_Barrel_[5]  = ElecGSFTrackResol;
      Var_wGwGSF_Barrel_[6]  = ElecGSFTracklnPt;
      Var_wGwGSF_Barrel_[7]  = ElecGSFTrackEta;
      Var_wGwGSF_Barrel_[8]  = TauEtaAtEcalEntrance;
      Var_wGwGSF_Barrel_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_wGwGSF_Barrel_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_wGwGSF_Barrel_[11] = std::log(std::max(float(1.), TauPt));
      Var_wGwGSF_Barrel_[12] = TauEmFraction;
      Var_wGwGSF_Barrel_[13] = TauSignalPFGammaCands;
      Var_wGwGSF_Barrel_[14] = TauLeadPFChargedHadrHoP;
      Var_wGwGSF_Barrel_[15] = TauLeadPFChargedHadrEoP;
      Var_wGwGSF_Barrel_[16] = TauVisMass;
      Var_wGwGSF_Barrel_[17] = TauHadrMva;
      Var_wGwGSF_Barrel_[18] = TauGammaEtaMom;
      Var_wGwGSF_Barrel_[19] = TauGammaPhiMom;
      Var_wGwGSF_Barrel_[20] = TauGammaEnFrac;
      Var_wGwGSF_Barrel_[21] = TauGSFChi2;
      Var_wGwGSF_Barrel_[22] = TauNumHitsVariable;
      Var_wGwGSF_Barrel_[23] = TauGSFTrackResol;
      Var_wGwGSF_Barrel_[24] = TauGSFTracklnPt;
      Var_wGwGSF_Barrel_[25] = TauGSFTrackEta;
      Var_wGwGSF_Barrel_[26] = TaudCrackEta;
      Var_wGwGSF_Barrel_[27] = TaudCrackPhi;
      mvaValue = mva_wGwGSF_BL_->GetClassifier(Var_wGwGSF_Barrel_);
    } else {
      Var_wGwGSF_Endcap_[0]  = ElecEtotOverPin;
      Var_wGwGSF_Endcap_[1]  = ElecEgammaOverPdif;
      Var_wGwGSF_Endcap_[2]  = ElecFbrem;
      Var_wGwGSF_Endcap_[3]  = ElecChi2GSF;
      Var_wGwGSF_Endcap_[4]  = ElecGSFNumHits;
      Var_wGwGSF_Endcap_[5]  = ElecGSFTrackResol;
      Var_wGwGSF_Endcap_[6]  = ElecGSFTracklnPt;
      Var_wGwGSF_Endcap_[7]  = ElecGSFTrackEta;
      Var_wGwGSF_Endcap_[8]  = TauEtaAtEcalEntrance;
      Var_wGwGSF_Endcap_[9]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_wGwGSF_Endcap_[10] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_wGwGSF_Endcap_[11] = std::log(std::max(float(1.), TauPt));
      Var_wGwGSF_Endcap_[12] = TauEmFraction;
      Var_wGwGSF_Endcap_[13] = TauSignalPFGammaCands;
      Var_wGwGSF_Endcap_[14] = TauLeadPFChargedHadrHoP;
      Var_wGwGSF_Endcap_[15] = TauLeadPFChargedHadrEoP;
      Var_wGwGSF_Endcap_[16] = TauVisMass;
      Var_wGwGSF_Endcap_[17] = TauHadrMva;
      Var_wGwGSF_Endcap_[18] = TauGammaEtaMom;
      Var_wGwGSF_Endcap_[19] = TauGammaPhiMom;
      Var_wGwGSF_Endcap_[20] = TauGammaEnFrac;
      Var_wGwGSF_Endcap_[21] = TauGSFChi2;
      Var_wGwGSF_Endcap_[22] = TauNumHitsVariable;
      Var_wGwGSF_Endcap_[23] = TauGSFTrackResol;
      Var_wGwGSF_Endcap_[24] = TauGSFTracklnPt;
      Var_wGwGSF_Endcap_[25] = TauGSFTrackEta;
      Var_wGwGSF_Endcap_[26] = TaudCrackEta;
      mvaValue = mva_wGwGSF_EC_->GetClassifier(Var_wGwGSF_Endcap_);
    } 
  }
  return mvaValue;
}

double AntiElectronIDMVA5::MVAValue(const reco::PFTau& thePFTau,
				    const reco::GsfElectron& theGsfEle)

{
  Float_t TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTau.signalPFCands();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    sumEtaTimesEnergy += (*pfCandidate)->positionAtECALEntrance().eta()*(*pfCandidate)->energy();
    sumEnergy += (*pfCandidate)->energy();
  }
  if ( sumEnergy > 0. ) {
    TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  }
  
  float TauLeadChargedPFCandEtaAtEcalEntrance = -99.;
  float TauLeadChargedPFCandPt = -99.;
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    const reco::Track* track = 0;
    if ( (*pfCandidate)->trackRef().isNonnull() ) track = (*pfCandidate)->trackRef().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->innerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->innerTrack().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->globalTrack().isNonnull() ) track = (*pfCandidate)->muonRef()->globalTrack().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->outerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->outerTrack().get();
    else if ( (*pfCandidate)->gsfTrackRef().isNonnull() ) track = (*pfCandidate)->gsfTrackRef().get();
    if ( track ) {
      if ( track->pt() > TauLeadChargedPFCandPt ) {
	TauLeadChargedPFCandEtaAtEcalEntrance = (*pfCandidate)->positionAtECALEntrance().eta();
	TauLeadChargedPFCandPt = track->pt();
      }
    }
  }

  Float_t TauPt = thePFTau.pt();
  Float_t TauEmFraction = std::max(thePFTau.emFraction(), (Float_t)0.);
  Float_t TauSignalPFGammaCands = thePFTau.signalPFGammaCands().size();
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }
  Float_t TauVisMass = thePFTau.mass();
  Float_t TauHadrMva = std::max(thePFTau.electronPreIDOutput(), float(-1.0));
  std::vector<Float_t> GammasdEta;
  std::vector<Float_t> GammasdPhi;
  std::vector<Float_t> GammasPt;
  for ( unsigned i = 0 ; i < thePFTau.signalPFGammaCands().size(); ++i ) {
    reco::PFCandidatePtr gamma = thePFTau.signalPFGammaCands().at(i);
    if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
      GammasdEta.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
    } else {
      GammasdEta.push_back(gamma->eta() - thePFTau.eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.phi());
    }
    GammasPt.push_back(gamma->pt());
  }
  Float_t TauKFNumHits = -99.;
  if ( (thePFTau.leadPFChargedHadrCand()->trackRef()).isNonnull() ) {
    TauKFNumHits = thePFTau.leadPFChargedHadrCand()->trackRef()->numberOfValidHits();
  }
  Float_t TauGSFNumHits = -99.;
  Float_t TauGSFChi2 = -99.;
  Float_t TauGSFTrackResol = -99.;
  Float_t TauGSFTracklnPt = -99.;
  Float_t TauGSFTrackEta = -99.;
  if ( (thePFTau.leadPFChargedHadrCand()->gsfTrackRef()).isNonnull() ) {
      TauGSFChi2 = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->normalizedChi2();
      TauGSFNumHits = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->numberOfValidHits();
      if ( thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt() > 0. ) {
	TauGSFTrackResol = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->ptError()/thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt();
	TauGSFTracklnPt = log(thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt())*M_LN10;
      }
      TauGSFTrackEta = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->eta();
  }
  Float_t TauPhi = thePFTau.phi();
  float sumPhiTimesEnergy = 0.;
  float sumEnergyPhi = 0.;
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    sumPhiTimesEnergy += (*pfCandidate)->positionAtECALEntrance().phi()*(*pfCandidate)->energy();
    sumEnergyPhi += (*pfCandidate)->energy();
  }
  if ( sumEnergyPhi > 0. ) {
    TauPhi = sumPhiTimesEnergy/sumEnergyPhi;
  }
  Float_t TaudCrackPhi = dCrackPhi(TauPhi, TauEtaAtEcalEntrance);
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance);
  Float_t TauSignalPFChargedCands = thePFTau.signalPFChargedHadrCands().size();
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();

  Float_t ElecEta = theGsfEle.eta();
  Float_t ElecPhi = theGsfEle.phi();
  Float_t ElecPt = theGsfEle.pt();
  //Variables related to the electron Cluster
  Float_t ElecEe = 0.;
  Float_t ElecEgamma = 0.;
  reco::SuperClusterRef pfSuperCluster = theGsfEle.parentSuperCluster();
  if ( pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable() ) {
    for ( reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin();
	  pfCluster != pfSuperCluster->clustersEnd(); ++pfCluster ) {
      double pfClusterEn = (*pfCluster)->energy();
      if ( pfCluster == pfSuperCluster->clustersBegin() ) ElecEe += pfClusterEn;
      else ElecEgamma += pfClusterEn;
    }
  }
  Float_t ElecPin = std::sqrt(theGsfEle.trackMomentumAtVtx().Mag2());
  Float_t ElecPout = std::sqrt(theGsfEle.trackMomentumOut().Mag2());
  Float_t ElecFbrem = theGsfEle.fbrem();
  //Variables related to the GsfTrack
  Float_t ElecChi2GSF = -99.;
  Float_t ElecGSFNumHits = -99.;
  Float_t ElecGSFTrackResol = -99.;
  Float_t ElecGSFTracklnPt = -99.;
  Float_t ElecGSFTrackEta = -99.;
  if ( theGsfEle.gsfTrack().isNonnull() ) {
    ElecChi2GSF = (theGsfEle).gsfTrack()->normalizedChi2();
    ElecGSFNumHits = (theGsfEle).gsfTrack()->numberOfValidHits();
    if ( theGsfEle.gsfTrack()->pt() > 0. ) {
      ElecGSFTrackResol = theGsfEle.gsfTrack()->ptError()/theGsfEle.gsfTrack()->pt();
      ElecGSFTracklnPt = log(theGsfEle.gsfTrack()->pt())*M_LN10;
    }
    ElecGSFTrackEta = theGsfEle.gsfTrack()->eta();
  }

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPt,
		  TauLeadChargedPFCandEtaAtEcalEntrance,
		  TauLeadChargedPFCandPt,
		  TaudCrackEta,
		  TaudCrackPhi,
		  TauEmFraction,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauVisMass,
		  TauHadrMva,
		  GammasdEta,
		  GammasdPhi,
		  GammasPt,
		  TauKFNumHits,				   
		  TauGSFNumHits,				   
		  TauGSFChi2,				   
		  TauGSFTrackResol,
		  TauGSFTracklnPt,
		  TauGSFTrackEta,
		  TauPhi,
		  TauSignalPFChargedCands,
		  TauHasGsf,
		  ElecEta,
		  ElecPhi,
		  ElecPt,
		  ElecEe,
		  ElecEgamma,
		  ElecPin,
		  ElecPout,
		  ElecFbrem,
		  ElecChi2GSF,
		  ElecGSFNumHits,
		  ElecGSFTrackResol,
		  ElecGSFTracklnPt,
		  ElecGSFTrackEta);
}

double AntiElectronIDMVA5::MVAValue(const reco::PFTau& thePFTau)
{
  Float_t TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTau.signalPFCands();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    sumEtaTimesEnergy += (*pfCandidate)->positionAtECALEntrance().eta()*(*pfCandidate)->energy();
    sumEnergy += (*pfCandidate)->energy();
  }
  if ( sumEnergy > 0. ) {
    TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  }
  
  float TauLeadChargedPFCandEtaAtEcalEntrance = -99.;
  float TauLeadChargedPFCandPt = -99.;
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    const reco::Track* track = 0;
    if ( (*pfCandidate)->trackRef().isNonnull() ) track = (*pfCandidate)->trackRef().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->innerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->innerTrack().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->globalTrack().isNonnull() ) track = (*pfCandidate)->muonRef()->globalTrack().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->outerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->outerTrack().get();
    else if ( (*pfCandidate)->gsfTrackRef().isNonnull() ) track = (*pfCandidate)->gsfTrackRef().get();
    if ( track ) {
      if ( track->pt() > TauLeadChargedPFCandPt ) {
	TauLeadChargedPFCandEtaAtEcalEntrance = (*pfCandidate)->positionAtECALEntrance().eta();
	TauLeadChargedPFCandPt = track->pt();
      }
    }
  }
  
  Float_t TauPt = thePFTau.pt();
  Float_t TauEmFraction = std::max(thePFTau.emFraction(), (Float_t)0.);
  Float_t TauSignalPFGammaCands = thePFTau.signalPFGammaCands().size();
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }
  Float_t TauVisMass = thePFTau.mass();
  Float_t TauHadrMva = std::max(thePFTau.electronPreIDOutput(),float(-1.0));
  std::vector<Float_t> GammasdEta;
  std::vector<Float_t> GammasdPhi;
  std::vector<Float_t> GammasPt;
  for ( unsigned i = 0 ; i < thePFTau.signalPFGammaCands().size(); ++i ) {
    reco::PFCandidatePtr gamma = thePFTau.signalPFGammaCands().at(i);
    if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
      GammasdEta.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
    } else {
      GammasdEta.push_back(gamma->eta() - thePFTau.eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.phi());
    }
    GammasPt.push_back(gamma->pt());
  }
  Float_t TauKFNumHits = -99.;
  if ( (thePFTau.leadPFChargedHadrCand()->trackRef()).isNonnull() ) {
    TauKFNumHits = thePFTau.leadPFChargedHadrCand()->trackRef()->numberOfValidHits();
  }
  Float_t TauGSFNumHits = -99.;
  Float_t TauGSFChi2 = -99.;
  Float_t TauGSFTrackResol = -99.;
  Float_t TauGSFTracklnPt = -99.;
  Float_t TauGSFTrackEta = -99.;
  if ( (thePFTau.leadPFChargedHadrCand()->gsfTrackRef()).isNonnull() ) {
    TauGSFChi2 = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->normalizedChi2();
    TauGSFNumHits = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->numberOfValidHits();
    if ( thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt() > 0. ) {
      TauGSFTrackResol = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->ptError()/thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt();
      TauGSFTracklnPt = log(thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt())*M_LN10;
    }
    TauGSFTrackEta = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->eta();
  }
  Float_t TauPhi = thePFTau.phi();
  float sumPhiTimesEnergy = 0.;
  float sumEnergyPhi = 0.;
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    sumPhiTimesEnergy += (*pfCandidate)->positionAtECALEntrance().phi()*(*pfCandidate)->energy();
    sumEnergyPhi += (*pfCandidate)->energy();
  }
  if ( sumEnergyPhi > 0. ) {
    TauPhi = sumPhiTimesEnergy/sumEnergyPhi;
  }
  Float_t TaudCrackPhi = dCrackPhi(TauPhi,TauEtaAtEcalEntrance) ;
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance) ;
  Float_t TauSignalPFChargedCands = thePFTau.signalPFChargedHadrCands().size();
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();

  Float_t dummyElecEta = 9.9;

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPt,
		  TauLeadChargedPFCandEtaAtEcalEntrance,
		  TauLeadChargedPFCandPt,
		  TaudCrackEta,
		  TaudCrackPhi,
		  TauEmFraction,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauVisMass,
		  TauHadrMva,
		  GammasdEta,
		  GammasdPhi,
		  GammasPt,
		  TauKFNumHits,				   
		  TauGSFNumHits,				   
		  TauGSFChi2,				   
		  TauGSFTrackResol,
		  TauGSFTracklnPt,
		  TauGSFTrackEta,
		  TauPhi,
		  TauSignalPFChargedCands,
		  TauHasGsf,
		  dummyElecEta,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.);
}

double AntiElectronIDMVA5::minimum(double a, double b)
{
  if ( std::abs(b) < std::abs(a) ) return b;
  else return a;
}


#include<array>
namespace {

  // IN: define locations of the 18 phi-cracks
 std::array<double,18> fill_cPhi() {
   constexpr double pi = M_PI; // 3.14159265358979323846;
   std::array<double,18> cPhi;
   // IN: define locations of the 18 phi-cracks
   cPhi[0] = 2.97025;
   for ( unsigned iCrack = 1; iCrack <= 17; ++iCrack )
      cPhi[iCrack] = cPhi[0] - 2.*iCrack*pi/18;
  return cPhi;
 }
     
  static const std::array<double,18> cPhi = fill_cPhi();

}



double AntiElectronIDMVA5::dCrackPhi(double phi, double eta)
{
//--- compute the (unsigned) distance to the closest phi-crack in the ECAL barrel  

  constexpr double pi = M_PI; // 3.14159265358979323846;

  // IN: shift of this location if eta < 0
  constexpr double delta_cPhi = 0.00638;

  double retVal = 99.; 

  if ( eta >= -1.47464 && eta <= 1.47464 ) {

    // the location is shifted
    if ( eta < 0. ) phi += delta_cPhi;

    // CV: need to bring-back phi into interval [-pi,+pi]
    if ( phi >  pi ) phi -= 2.*pi;
    if ( phi < -pi ) phi += 2.*pi;

    if ( phi >= -pi && phi <= pi ) {

      // the problem of the extrema:
      if ( phi < cPhi[17] || phi >= cPhi[0] ) {
	if ( phi < 0. ) phi += 2.*pi;
	retVal = minimum(phi - cPhi[0], phi - cPhi[17] - 2.*pi);        	
      } else {
	// between these extrema...
	bool OK = false;
	unsigned iCrack = 16;
	while( !OK ) {
	  if ( phi < cPhi[iCrack] ) {
	    retVal = minimum(phi - cPhi[iCrack + 1], phi - cPhi[iCrack]);
	    OK = true;
	  } else {
	    iCrack -= 1;
	  }
	}
      }
    } else {
      retVal = 0.; // IN: if there is a problem, we assume that we are in a crack
    }
  } else {
    return -99.;       
  }
  
  return std::abs(retVal);
}

double AntiElectronIDMVA5::dCrackEta(double eta)
{
//--- compute the (unsigned) distance to the closest eta-crack in the ECAL barrel
  
  // IN: define locations of the eta-cracks
  double cracks[5] = { 0., 4.44747e-01, 7.92824e-01, 1.14090e+00, 1.47464e+00 };
  
  double retVal = 99.;
  
  for ( int iCrack = 0; iCrack < 5 ; ++iCrack ) {
    double d = minimum(eta - cracks[iCrack], eta + cracks[iCrack]);
    if ( std::abs(d) < std::abs(retVal) ) {
      retVal = d;
    }
  }

  return std::abs(retVal);
}
