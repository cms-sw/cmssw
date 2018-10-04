#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <TMath.h>
#include <TFile.h>
#include <array>

AntiElectronIDMVA6::AntiElectronIDMVA6(const edm::ParameterSet& cfg)
  : isInitialized_(false),
    mva_NoEleMatch_woGwoGSF_BL_(nullptr),
    mva_NoEleMatch_wGwoGSF_BL_(nullptr),
    mva_woGwGSF_BL_(nullptr),
    mva_wGwGSF_BL_(nullptr),
    mva_NoEleMatch_woGwoGSF_EC_(nullptr),
    mva_NoEleMatch_wGwoGSF_EC_(nullptr),
    mva_woGwGSF_EC_(nullptr),
    mva_wGwGSF_EC_(nullptr)   
{
  loadMVAfromDB_ = cfg.exists("loadMVAfromDB") ? cfg.getParameter<bool>("loadMVAfromDB"): false;
  if ( !loadMVAfromDB_ ) {
    if(cfg.exists("inputFileName")){
      inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
    }else throw cms::Exception("MVA input not defined") << "Requested to load tau MVA input from ROOT file but no file provided in cfg file";
    
  }

  mvaName_NoEleMatch_woGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwoGSF_BL");
  mvaName_NoEleMatch_wGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwoGSF_BL");
  mvaName_woGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_woGwGSF_BL");
  mvaName_wGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_wGwGSF_BL");
  mvaName_NoEleMatch_woGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwoGSF_EC");
  mvaName_NoEleMatch_wGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwoGSF_EC");
  mvaName_woGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_woGwGSF_EC");
  mvaName_wGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_wGwGSF_EC");

  usePhiAtEcalEntranceExtrapolation_ = cfg.getParameter<bool>("usePhiAtEcalEntranceExtrapolation");

  Var_NoEleMatch_woGwoGSF_Barrel_ = new Float_t[10];
  Var_NoEleMatch_wGwoGSF_Barrel_ = new Float_t[18];
  Var_woGwGSF_Barrel_ = new Float_t[24];
  Var_wGwGSF_Barrel_ = new Float_t[32];
  Var_NoEleMatch_woGwoGSF_Endcap_ = new Float_t[9];
  Var_NoEleMatch_wGwoGSF_Endcap_ = new Float_t[17];
  Var_woGwGSF_Endcap_ = new Float_t[23];
  Var_wGwGSF_Endcap_ = new Float_t[31];

  bField_ = 0;
  verbosity_ = 0;
}

AntiElectronIDMVA6::~AntiElectronIDMVA6()
{
  delete [] Var_NoEleMatch_woGwoGSF_Barrel_;
  delete [] Var_NoEleMatch_wGwoGSF_Barrel_;
  delete [] Var_woGwGSF_Barrel_;
  delete [] Var_wGwGSF_Barrel_;
  delete [] Var_NoEleMatch_woGwoGSF_Endcap_;
  delete [] Var_NoEleMatch_wGwoGSF_Endcap_;
  delete [] Var_woGwGSF_Endcap_;
  delete [] Var_wGwGSF_Endcap_;

  if ( !loadMVAfromDB_ ){ 
    delete mva_NoEleMatch_woGwoGSF_BL_;
    delete mva_NoEleMatch_wGwoGSF_BL_;
    delete mva_woGwGSF_BL_;
    delete mva_wGwGSF_BL_;
    delete mva_NoEleMatch_woGwoGSF_EC_;
    delete mva_NoEleMatch_wGwoGSF_EC_;
    delete mva_woGwGSF_EC_;
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
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if ( !mva )
      throw cms::Exception("PFRecoTauDiscriminationAgainstElectronMVA6::loadMVA")
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

void AntiElectronIDMVA6::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( !isInitialized_ ) {
    if ( loadMVAfromDB_ ) {
      mva_NoEleMatch_woGwoGSF_BL_ = loadMVAfromDB(es, mvaName_NoEleMatch_woGwoGSF_BL_);
      mva_NoEleMatch_wGwoGSF_BL_  = loadMVAfromDB(es, mvaName_NoEleMatch_wGwoGSF_BL_);
      mva_woGwGSF_BL_             = loadMVAfromDB(es, mvaName_woGwGSF_BL_);
      mva_wGwGSF_BL_              = loadMVAfromDB(es, mvaName_wGwGSF_BL_);
      mva_NoEleMatch_woGwoGSF_EC_ = loadMVAfromDB(es, mvaName_NoEleMatch_woGwoGSF_EC_);
      mva_NoEleMatch_wGwoGSF_EC_  = loadMVAfromDB(es, mvaName_NoEleMatch_wGwoGSF_EC_);
      mva_woGwGSF_EC_             = loadMVAfromDB(es, mvaName_woGwGSF_EC_);
      mva_wGwGSF_EC_              = loadMVAfromDB(es, mvaName_wGwGSF_EC_);  
    } else {
          if ( inputFileName_.location() == edm::FileInPath::Unknown ) throw cms::Exception("PFRecoTauDiscriminationAgainstElectronMVA6::loadMVA")
          << " Failed to find File = " << inputFileName_ << " !!\n";
          TFile* inputFile = new TFile(inputFileName_.fullPath().data());

      mva_NoEleMatch_woGwoGSF_BL_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwoGSF_BL_);
      mva_NoEleMatch_wGwoGSF_BL_  = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwoGSF_BL_);
      mva_woGwGSF_BL_             = loadMVAfromFile(inputFile, mvaName_woGwGSF_BL_);
      mva_wGwGSF_BL_              = loadMVAfromFile(inputFile, mvaName_wGwGSF_BL_);
      mva_NoEleMatch_woGwoGSF_EC_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwoGSF_EC_);
      mva_NoEleMatch_wGwoGSF_EC_  = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwoGSF_EC_);
      mva_woGwGSF_EC_             = loadMVAfromFile(inputFile, mvaName_woGwGSF_EC_);
      mva_wGwGSF_EC_              = loadMVAfromFile(inputFile, mvaName_wGwGSF_EC_);
      inputFilesToDelete_.push_back(inputFile);  
    }
    isInitialized_ = true;
  }

  edm::ESHandle<MagneticField> pSetup;
  es.get<IdealMagneticFieldRecord>().get(pSetup);
  bField_ = pSetup->inTesla(GlobalPoint(0,0,0)).z();
}

double AntiElectronIDMVA6::MVAValue(Float_t TauPt,
                                    Float_t TauEtaAtEcalEntrance,
                                    Float_t TauPhi,
                                    Float_t TauLeadChargedPFCandPt,
                                    Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
                                    Float_t TauEmFraction,
                                    Float_t TauLeadPFChargedHadrHoP,
                                    Float_t TauLeadPFChargedHadrEoP,
                                    Float_t TauVisMassIn,
                                    Float_t TaudCrackEta,
                                    Float_t TaudCrackPhi,
                                    Float_t TauHasGsf,
                                    Int_t TauSignalPFGammaCandsIn,
                                    Int_t TauSignalPFGammaCandsOut,
                                    const std::vector<Float_t>& GammasdEtaInSigCone,
                                    const std::vector<Float_t>& GammasdPhiInSigCone,
                                    const std::vector<Float_t>& GammasPtInSigCone,
                                    const std::vector<Float_t>& GammasdEtaOutSigCone,
                                    const std::vector<Float_t>& GammasdPhiOutSigCone,
                                    const std::vector<Float_t>& GammasPtOutSigCone,
                                    Float_t ElecEta,
                                    Float_t ElecPhi,
                                    Float_t ElecEtotOverPin,
                                    Float_t ElecChi2NormGSF,
                                    Float_t ElecChi2NormKF,
                                    Float_t ElecGSFNumHits,
                                    Float_t ElecKFNumHits,
                                    Float_t ElecGSFTrackResol,
                                    Float_t ElecGSFTracklnPt,
                                    Float_t ElecPin,
                                    Float_t ElecPout,
                                    Float_t ElecEecal,
                                    Float_t ElecDeltaEta,
                                    Float_t ElecDeltaPhi,
                                    Float_t ElecMvaInSigmaEtaEta,
                                    Float_t ElecMvaInHadEnergy,
                                    Float_t ElecMvaInDeltaEta)
{ 
  double sumPt  = 0.;
  double dEta2  = 0.;
  double dPhi2  = 0.;
  double sumPt2 = 0.;
  for ( unsigned int i = 0 ; i < GammasPtInSigCone.size() ; ++i ) {
    double pt_i  = GammasPtInSigCone[i];
    double phi_i = GammasdPhiInSigCone[i];
    if ( GammasdPhiInSigCone[i] > M_PI ) phi_i = GammasdPhiInSigCone[i] - 2*M_PI;
    else if ( GammasdPhiInSigCone[i] < -M_PI ) phi_i = GammasdPhiInSigCone[i] + 2*M_PI;
    double eta_i = GammasdEtaInSigCone[i];
    sumPt  +=  pt_i;
    sumPt2 += (pt_i*pt_i);
    dEta2  += (pt_i*eta_i*eta_i);
    dPhi2  += (pt_i*phi_i*phi_i);
  }
  Float_t TauGammaEnFracIn = -99.;
  if ( TauPt > 0. ) {
    TauGammaEnFracIn = sumPt/TauPt;
  }
  if ( sumPt > 0. ) {
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }
  Float_t TauGammaEtaMomIn = std::sqrt(dEta2)*std::sqrt(TauGammaEnFracIn)*TauPt;
  Float_t TauGammaPhiMomIn = std::sqrt(dPhi2)*std::sqrt(TauGammaEnFracIn)*TauPt;

  sumPt  = 0.;
  dEta2  = 0.;
  dPhi2  = 0.;
  sumPt2 = 0.;
  for ( unsigned int i = 0 ; i < GammasPtOutSigCone.size() ; ++i ) {
    double pt_i  = GammasPtOutSigCone[i];
    double phi_i = GammasdPhiOutSigCone[i];
    if ( GammasdPhiOutSigCone[i] > M_PI ) phi_i = GammasdPhiOutSigCone[i] - 2*M_PI;
    else if ( GammasdPhiOutSigCone[i] < -M_PI ) phi_i = GammasdPhiOutSigCone[i] + 2*M_PI;
    double eta_i = GammasdEtaOutSigCone[i];
    sumPt  +=  pt_i;
    sumPt2 += (pt_i*pt_i);
    dEta2  += (pt_i*eta_i*eta_i);
    dPhi2  += (pt_i*phi_i*phi_i);
  }
  Float_t TauGammaEnFracOut = sumPt/TauPt;
  if ( sumPt > 0. ) {
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }
  Float_t TauGammaEtaMomOut = std::sqrt(dEta2)*std::sqrt(TauGammaEnFracOut)*TauPt;
  Float_t TauGammaPhiMomOut = std::sqrt(dPhi2)*std::sqrt(TauGammaEnFracOut)*TauPt;
  
  return MVAValue(TauPt,
                  TauEtaAtEcalEntrance,
                  TauPhi,
                  TauLeadChargedPFCandPt,
                  TauLeadChargedPFCandEtaAtEcalEntrance,
                  TauEmFraction,
                  TauLeadPFChargedHadrHoP,
                  TauLeadPFChargedHadrEoP,
                  TauVisMassIn,
                  TaudCrackEta,
                  TaudCrackPhi,
                  TauHasGsf,
                  TauSignalPFGammaCandsIn,
                  TauSignalPFGammaCandsOut,
                  TauGammaEtaMomIn,
                  TauGammaEtaMomOut,
                  TauGammaPhiMomIn,
                  TauGammaPhiMomOut,
                  TauGammaEnFracIn,
                  TauGammaEnFracOut,
                  ElecEta,
                  ElecPhi,
                  ElecEtotOverPin,
                  ElecChi2NormGSF,
                  ElecChi2NormKF,
                  ElecGSFNumHits,
                  ElecKFNumHits,
                  ElecGSFTrackResol,
                  ElecGSFTracklnPt,
                  ElecPin,
                  ElecPout,
                  ElecEecal,
                  ElecDeltaEta,
                  ElecDeltaPhi,
                  ElecMvaInSigmaEtaEta,
                  ElecMvaInHadEnergy,
                  ElecMvaInDeltaEta);
}

double AntiElectronIDMVA6::MVAValue(Float_t TauPt,
                                    Float_t TauEtaAtEcalEntrance,
                                    Float_t TauPhi,
                                    Float_t TauLeadChargedPFCandPt,
                                    Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
                                    Float_t TauEmFraction,
                                    Float_t TauLeadPFChargedHadrHoP,
                                    Float_t TauLeadPFChargedHadrEoP,
                                    Float_t TauVisMassIn,
                                    Float_t TaudCrackEta,
                                    Float_t TaudCrackPhi,
                                    Float_t TauHasGsf,
                                    Int_t TauSignalPFGammaCandsIn,
                                    Int_t TauSignalPFGammaCandsOut,
                                    Float_t TauGammaEtaMomIn,
                                    Float_t TauGammaEtaMomOut,
                                    Float_t TauGammaPhiMomIn,
                                    Float_t TauGammaPhiMomOut,
                                    Float_t TauGammaEnFracIn,
                                    Float_t TauGammaEnFracOut,
                                    Float_t ElecEta,
                                    Float_t ElecPhi,
                                    Float_t ElecEtotOverPin,
                                    Float_t ElecChi2NormGSF,
                                    Float_t ElecChi2NormKF,
                                    Float_t ElecGSFNumHits,
                                    Float_t ElecKFNumHits,
                                    Float_t ElecGSFTrackResol,
                                    Float_t ElecGSFTracklnPt,
                                    Float_t ElecPin,
                                    Float_t ElecPout,
                                    Float_t ElecEecal,
                                    Float_t ElecDeltaEta,
                                    Float_t ElecDeltaPhi,
                                    Float_t ElecMvaInSigmaEtaEta,
                                    Float_t ElecMvaInHadEnergy,
                                    Float_t ElecMvaInDeltaEta)
{

  if ( !isInitialized_ ) {
    throw cms::Exception("ClassNotInitialized")
      << " AntiElectronMVA not properly initialized !!\n";
  }

  double mvaValue = -99.;
  
  const float ECALBarrelEndcapEtaBorder = 1.479;
  float ElecDeltaPinPoutOverPin = (ElecPin > 0.0) ? (std::abs(ElecPin - ElecPout)/ElecPin) : 1.0;
  float ElecEecalOverPout = (ElecPout > 0.0) ? (ElecEecal/ElecPout) : 20.0;
  float ElecNumHitsDiffOverSum = ((ElecGSFNumHits + ElecKFNumHits) > 0.0) ? 
                                 ((ElecGSFNumHits - ElecKFNumHits)/(ElecGSFNumHits + ElecKFNumHits)) : 1.0;
  
  if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 && TauSignalPFGammaCandsIn == 0 && TauHasGsf < 0.5) {
    if ( std::abs(TauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ){    
      Var_NoEleMatch_woGwoGSF_Barrel_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[1] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[2] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwoGSF_Barrel_[3] = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_woGwoGSF_Barrel_[4] = TauEmFraction;
      Var_NoEleMatch_woGwoGSF_Barrel_[5] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[6] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[7] = TauVisMassIn;
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
      Var_NoEleMatch_woGwoGSF_Endcap_[7] = TauVisMassIn;
      Var_NoEleMatch_woGwoGSF_Endcap_[8] = TaudCrackEta;
      mvaValue = mva_NoEleMatch_woGwoGSF_EC_->GetClassifier(Var_NoEleMatch_woGwoGSF_Endcap_);
    }
  }
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 && TauSignalPFGammaCandsIn > 0 && TauHasGsf < 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ){
      Var_NoEleMatch_wGwoGSF_Barrel_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Barrel_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Barrel_[4]  = TauEmFraction;
      Var_NoEleMatch_wGwoGSF_Barrel_[5]  = TauSignalPFGammaCandsIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[6]  = TauSignalPFGammaCandsOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[7]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[8]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[9]  = TauVisMassIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[10] = TauGammaEtaMomIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[11] = TauGammaEtaMomOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[12] = TauGammaPhiMomIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[13] = TauGammaPhiMomOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[14] = TauGammaEnFracIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[15] = TauGammaEnFracOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[16] = TaudCrackEta;
      Var_NoEleMatch_wGwoGSF_Barrel_[17] = TaudCrackPhi;
      mvaValue = mva_NoEleMatch_wGwoGSF_BL_->GetClassifier(Var_NoEleMatch_wGwoGSF_Barrel_);
    } else {
      Var_NoEleMatch_wGwoGSF_Endcap_[0]  = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[1]  = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[2]  = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Endcap_[3]  = std::log(std::max(float(1.), TauPt));
      Var_NoEleMatch_wGwoGSF_Endcap_[4]  = TauEmFraction;
      Var_NoEleMatch_wGwoGSF_Endcap_[5]  = TauSignalPFGammaCandsIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[6]  = TauSignalPFGammaCandsOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[7]  = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[8]  = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[9]  = TauVisMassIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[10] = TauGammaEtaMomIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[11] = TauGammaEtaMomOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[12] = TauGammaPhiMomIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[13] = TauGammaPhiMomOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[14] = TauGammaEnFracIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[15] = TauGammaEnFracOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[16] = TaudCrackEta;
      mvaValue = mva_NoEleMatch_wGwoGSF_EC_->GetClassifier(Var_NoEleMatch_wGwoGSF_Endcap_);
    }
  }    
  else if ( TauSignalPFGammaCandsIn == 0 && TauHasGsf > 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ) {
      Var_woGwGSF_Barrel_[0]  = std::max(float(-0.1), ElecEtotOverPin);
      Var_woGwGSF_Barrel_[1]  = std::log(std::max(float(0.01), ElecChi2NormGSF));
      Var_woGwGSF_Barrel_[2]  = ElecGSFNumHits;
      Var_woGwGSF_Barrel_[3]  = std::log(std::max(float(0.01), ElecGSFTrackResol));
      Var_woGwGSF_Barrel_[4]  = ElecGSFTracklnPt;
      Var_woGwGSF_Barrel_[5]  = ElecNumHitsDiffOverSum;
      Var_woGwGSF_Barrel_[6]  = std::log(std::max(float(0.01), ElecChi2NormKF));
      Var_woGwGSF_Barrel_[7]  = std::min(ElecDeltaPinPoutOverPin, float(1.));
      Var_woGwGSF_Barrel_[8]  = std::min(ElecEecalOverPout, float(20.));
      Var_woGwGSF_Barrel_[9]  = ElecDeltaEta;
      Var_woGwGSF_Barrel_[10] = ElecDeltaPhi;
      Var_woGwGSF_Barrel_[11] = std::min(ElecMvaInSigmaEtaEta, float(0.01));
      Var_woGwGSF_Barrel_[12] = std::min(ElecMvaInHadEnergy, float(20.));
      Var_woGwGSF_Barrel_[13] = std::min(ElecMvaInDeltaEta, float(0.1));
      Var_woGwGSF_Barrel_[14] = TauEtaAtEcalEntrance;
      Var_woGwGSF_Barrel_[15] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_woGwGSF_Barrel_[16] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_woGwGSF_Barrel_[17] = std::log(std::max(float(1.), TauPt));
      Var_woGwGSF_Barrel_[18] = TauEmFraction;
      Var_woGwGSF_Barrel_[19] = TauLeadPFChargedHadrHoP;
      Var_woGwGSF_Barrel_[20] = TauLeadPFChargedHadrEoP;
      Var_woGwGSF_Barrel_[21] = TauVisMassIn;
      Var_woGwGSF_Barrel_[22] = TaudCrackEta;
      Var_woGwGSF_Barrel_[23] = TaudCrackPhi;
      mvaValue = mva_woGwGSF_BL_->GetClassifier(Var_woGwGSF_Barrel_);
    } else {
      Var_woGwGSF_Endcap_[0]  = std::max(float(-0.1), ElecEtotOverPin);
      Var_woGwGSF_Endcap_[1]  = std::log(std::max(float(0.01), ElecChi2NormGSF));
      Var_woGwGSF_Endcap_[2]  = ElecGSFNumHits;
      Var_woGwGSF_Endcap_[3]  = std::log(std::max(float(0.01), ElecGSFTrackResol));
      Var_woGwGSF_Endcap_[4]  = ElecGSFTracklnPt;
      Var_woGwGSF_Endcap_[5]  = ElecNumHitsDiffOverSum;
      Var_woGwGSF_Endcap_[6]  = std::log(std::max(float(0.01), ElecChi2NormKF));
      Var_woGwGSF_Endcap_[7]  = std::min(ElecDeltaPinPoutOverPin, float(1.));
      Var_woGwGSF_Endcap_[8]  = std::min(ElecEecalOverPout, float(20.));
      Var_woGwGSF_Endcap_[9]  = ElecDeltaEta;
      Var_woGwGSF_Endcap_[10] = ElecDeltaPhi;
      Var_woGwGSF_Endcap_[11] = std::min(ElecMvaInSigmaEtaEta, float(0.01));
      Var_woGwGSF_Endcap_[12] = std::min(ElecMvaInHadEnergy, float(20.));
      Var_woGwGSF_Endcap_[13] = std::min(ElecMvaInDeltaEta, float(0.1));
      Var_woGwGSF_Endcap_[14] = TauEtaAtEcalEntrance;
      Var_woGwGSF_Endcap_[15] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_woGwGSF_Endcap_[16] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_woGwGSF_Endcap_[17] = std::log(std::max(float(1.), TauPt));
      Var_woGwGSF_Endcap_[18] = TauEmFraction;
      Var_woGwGSF_Endcap_[19] = TauLeadPFChargedHadrHoP;
      Var_woGwGSF_Endcap_[20] = TauLeadPFChargedHadrEoP;
      Var_woGwGSF_Endcap_[21] = TauVisMassIn;
      Var_woGwGSF_Endcap_[22] = TaudCrackEta;
      mvaValue = mva_woGwGSF_EC_->GetClassifier(Var_woGwGSF_Endcap_);
    } 
  }
  else if ( TauSignalPFGammaCandsIn > 0 && TauHasGsf > 0.5 ) {
    if ( std::abs(TauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ) {
      Var_wGwGSF_Barrel_[0]  = std::max(float(-0.1), ElecEtotOverPin);
      Var_wGwGSF_Barrel_[1]  = std::log(std::max(float(0.01), ElecChi2NormGSF));
      Var_wGwGSF_Barrel_[2]  = ElecGSFNumHits;
      Var_wGwGSF_Barrel_[3]  = std::log(std::max(float(0.01), ElecGSFTrackResol));
      Var_wGwGSF_Barrel_[4]  = ElecGSFTracklnPt;
      Var_wGwGSF_Barrel_[5]  = ElecNumHitsDiffOverSum;
      Var_wGwGSF_Barrel_[6]  = std::log(std::max(float(0.01), ElecChi2NormKF));
      Var_wGwGSF_Barrel_[7]  = std::min(ElecDeltaPinPoutOverPin, float(1.));
      Var_wGwGSF_Barrel_[8]  = std::min(ElecEecalOverPout, float(20.));
      Var_wGwGSF_Barrel_[9]  = ElecDeltaEta;
      Var_wGwGSF_Barrel_[10] = ElecDeltaPhi;
      Var_wGwGSF_Barrel_[11] = std::min(ElecMvaInSigmaEtaEta, float(0.01));
      Var_wGwGSF_Barrel_[12] = std::min(ElecMvaInHadEnergy, float(20.));
      Var_wGwGSF_Barrel_[13] = std::min(ElecMvaInDeltaEta, float(0.1));
      Var_wGwGSF_Barrel_[14] = TauEtaAtEcalEntrance;
      Var_wGwGSF_Barrel_[15] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_wGwGSF_Barrel_[16] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_wGwGSF_Barrel_[17] = std::log(std::max(float(1.), TauPt));
      Var_wGwGSF_Barrel_[18] = TauEmFraction;
      Var_wGwGSF_Barrel_[19] = TauSignalPFGammaCandsIn;
      Var_wGwGSF_Barrel_[20] = TauSignalPFGammaCandsOut;
      Var_wGwGSF_Barrel_[21] = TauLeadPFChargedHadrHoP;
      Var_wGwGSF_Barrel_[22] = TauLeadPFChargedHadrEoP;
      Var_wGwGSF_Barrel_[23] = TauVisMassIn;
      Var_wGwGSF_Barrel_[24] = TauGammaEtaMomIn;
      Var_wGwGSF_Barrel_[25] = TauGammaEtaMomOut;
      Var_wGwGSF_Barrel_[26] = TauGammaPhiMomIn;
      Var_wGwGSF_Barrel_[27] = TauGammaPhiMomOut;
      Var_wGwGSF_Barrel_[28] = TauGammaEnFracIn;
      Var_wGwGSF_Barrel_[29] = TauGammaEnFracOut;
      Var_wGwGSF_Barrel_[30] = TaudCrackEta;
      Var_wGwGSF_Barrel_[31] = TaudCrackPhi;
      mvaValue = mva_wGwGSF_BL_->GetClassifier(Var_wGwGSF_Barrel_);
    } else {
      Var_wGwGSF_Endcap_[0]  = std::max(float(-0.1), ElecEtotOverPin);
      Var_wGwGSF_Endcap_[1]  = std::log(std::max(float(0.01), ElecChi2NormGSF));
      Var_wGwGSF_Endcap_[2]  = ElecGSFNumHits;
      Var_wGwGSF_Endcap_[3]  = std::log(std::max(float(0.01), ElecGSFTrackResol));
      Var_wGwGSF_Endcap_[4]  = ElecGSFTracklnPt;
      Var_wGwGSF_Endcap_[5]  = ElecNumHitsDiffOverSum;
      Var_wGwGSF_Endcap_[6]  = std::log(std::max(float(0.01), ElecChi2NormKF));
      Var_wGwGSF_Endcap_[7]  = std::min(ElecDeltaPinPoutOverPin, float(1.));
      Var_wGwGSF_Endcap_[8]  = std::min(ElecEecalOverPout, float(20.));
      Var_wGwGSF_Endcap_[9]  = ElecDeltaEta;
      Var_wGwGSF_Endcap_[10] = ElecDeltaPhi;
      Var_wGwGSF_Endcap_[11] = std::min(ElecMvaInSigmaEtaEta, float(0.01));
      Var_wGwGSF_Endcap_[12] = std::min(ElecMvaInHadEnergy, float(20.));
      Var_wGwGSF_Endcap_[13] = std::min(ElecMvaInDeltaEta, float(0.1));
      Var_wGwGSF_Endcap_[14] = TauEtaAtEcalEntrance;
      Var_wGwGSF_Endcap_[15] = TauLeadChargedPFCandEtaAtEcalEntrance;
      Var_wGwGSF_Endcap_[16] = std::min(float(2.), TauLeadChargedPFCandPt/std::max(float(1.), TauPt));
      Var_wGwGSF_Endcap_[17] = std::log(std::max(float(1.), TauPt));
      Var_wGwGSF_Endcap_[18] = TauEmFraction;
      Var_wGwGSF_Endcap_[19] = TauSignalPFGammaCandsIn;
      Var_wGwGSF_Endcap_[20] = TauSignalPFGammaCandsOut;
      Var_wGwGSF_Endcap_[21] = TauLeadPFChargedHadrHoP;
      Var_wGwGSF_Endcap_[22] = TauLeadPFChargedHadrEoP;
      Var_wGwGSF_Endcap_[23] = TauVisMassIn;
      Var_wGwGSF_Endcap_[24] = TauGammaEtaMomIn;
      Var_wGwGSF_Endcap_[25] = TauGammaEtaMomOut;
      Var_wGwGSF_Endcap_[26] = TauGammaPhiMomIn;
      Var_wGwGSF_Endcap_[27] = TauGammaPhiMomOut;
      Var_wGwGSF_Endcap_[28] = TauGammaEnFracIn;
      Var_wGwGSF_Endcap_[29] = TauGammaEnFracOut;
      Var_wGwGSF_Endcap_[30] = TaudCrackEta;
      mvaValue = mva_wGwGSF_EC_->GetClassifier(Var_wGwGSF_Endcap_);
    } 
  }
  return mvaValue;
}

double AntiElectronIDMVA6::MVAValue(const reco::PFTau& thePFTau,
				    const reco::GsfElectron& theGsfEle)

{
  // === tau variables ===
  float TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTau.signalPFCands();
  for ( const auto & pfCandidate : signalPFCands ) {
    sumEtaTimesEnergy += pfCandidate->positionAtECALEntrance().eta()*pfCandidate->energy();
    sumEnergy += pfCandidate->energy();
  }
  if ( sumEnergy > 0. ) {
    TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  }
  
  float TauLeadChargedPFCandEtaAtEcalEntrance = -99.;
  float TauLeadChargedPFCandPt = -99.;
  for ( const auto & pfCandidate : signalPFCands ) {
    const reco::Track* track = nullptr;
    if ( pfCandidate->trackRef().isNonnull() ) track = pfCandidate->trackRef().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->innerTrack().isNonnull()  ) track = pfCandidate->muonRef()->innerTrack().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->globalTrack().isNonnull() ) track = pfCandidate->muonRef()->globalTrack().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->outerTrack().isNonnull()  ) track = pfCandidate->muonRef()->outerTrack().get();
    else if ( pfCandidate->gsfTrackRef().isNonnull() ) track = pfCandidate->gsfTrackRef().get();
    if ( track ) {
      if ( track->pt() > TauLeadChargedPFCandPt ) {
	TauLeadChargedPFCandEtaAtEcalEntrance = pfCandidate->positionAtECALEntrance().eta();
	TauLeadChargedPFCandPt = track->pt();
      }
    }
  }

  Float_t TauPt = thePFTau.pt();
  Float_t TauEmFraction = std::max(thePFTau.emFraction(), (Float_t)0.);
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }

  std::vector<Float_t> GammasdEtaInSigCone;
  std::vector<Float_t> GammasdPhiInSigCone;
  std::vector<Float_t> GammasPtInSigCone;
  std::vector<Float_t> GammasdEtaOutSigCone;
  std::vector<Float_t> GammasdPhiOutSigCone;
  std::vector<Float_t> GammasPtOutSigCone;
  reco::Candidate::LorentzVector pfGammaSum(0,0,0,0);
  reco::Candidate::LorentzVector pfChargedSum(0,0,0,0);
  
  for ( const auto & gamma : thePFTau.signalPFGammaCands() ) {
    float dR = deltaR(gamma->p4(), thePFTau.leadPFChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, thePFTau.pt())));

    // pfGammas inside the tau signal cone
    if (dR < signalrad) {
      if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
        GammasdEtaInSigCone.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
      }
      else {
        GammasdEtaInSigCone.push_back(gamma->eta() - thePFTau.eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - thePFTau.phi());
      }
      GammasPtInSigCone.push_back(gamma->pt());
      pfGammaSum += gamma->p4();
    }
    // pfGammas outside the tau signal cone
    else {
      if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
        GammasdEtaOutSigCone.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
      } 
      else {
        GammasdEtaOutSigCone.push_back(gamma->eta() - thePFTau.eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - thePFTau.phi());
      }
      GammasPtOutSigCone.push_back(gamma->pt());
    }
  }
  
  for ( const auto & charged : thePFTau.signalPFChargedHadrCands() ) {
    float dR = deltaR(charged->p4(), thePFTau.leadPFChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, thePFTau.pt())));
  
    // charged particles inside the tau signal cone
    if (dR < signalrad) {
        pfChargedSum += charged->p4();
    }
  }
  
  Int_t TauSignalPFGammaCandsIn = GammasPtInSigCone.size();
  Int_t TauSignalPFGammaCandsOut = GammasPtOutSigCone.size();
  Float_t TauVisMassIn = (pfGammaSum + pfChargedSum).mass();

  Float_t TauPhi = thePFTau.phi();
  float sumPhiTimesEnergy = 0.;
  float sumEnergyPhi = 0.;
  if ( !usePhiAtEcalEntranceExtrapolation_ ) {
    for ( const auto & pfc : signalPFCands ) {
      sumPhiTimesEnergy += pfc->positionAtECALEntrance().phi()*pfc->energy();
      sumEnergyPhi += pfc->energy();
    }
  }
  else{
    TauPhi= -99.;
    for ( const auto & signalPFCand : signalPFCands ) {
      reco::Candidate const*  signalCand = signalPFCand.get();
      float phi = thePFTau.phi();
      math::XYZPoint aPos; 
      if ( atECalEntrance(signalCand, aPos) ) phi = aPos.Phi();
      sumPhiTimesEnergy += phi*signalCand->energy();     
      sumEnergy += signalCand->energy();
    }
  }
  if ( sumEnergyPhi > 0. ) {
    TauPhi = sumPhiTimesEnergy/sumEnergyPhi;
  }
  Float_t TaudCrackPhi = dCrackPhi(TauPhi, TauEtaAtEcalEntrance);
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance);
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();

  
  // === electron variables ===
  Float_t ElecEta = theGsfEle.eta();
  Float_t ElecPhi = theGsfEle.phi();
                  
  //Variables related to the electron Cluster
  Float_t ElecEe = 0.;
  Float_t ElecEgamma = 0.;
  reco::SuperClusterRef pfSuperCluster = theGsfEle.superCluster();
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
  Float_t ElecEtotOverPin = (ElecPin > 0.0) ? ((ElecEe + ElecEgamma)/ElecPin) : -0.1;
  Float_t ElecEecal = theGsfEle.ecalEnergy();
  Float_t ElecDeltaEta = theGsfEle.deltaEtaSeedClusterTrackAtCalo();
  Float_t ElecDeltaPhi = theGsfEle.deltaPhiSeedClusterTrackAtCalo();
  Float_t ElecMvaInSigmaEtaEta = (theGsfEle).mvaInput().sigmaEtaEta;
  Float_t ElecMvaInHadEnergy = (theGsfEle).mvaInput().hadEnergy;
  Float_t ElecMvaInDeltaEta = (theGsfEle).mvaInput().deltaEta;
  
  //Variables related to the GsfTrack
  Float_t ElecChi2NormGSF = -99.;
  Float_t ElecGSFNumHits = -99.;
  Float_t ElecGSFTrackResol = -99.;
  Float_t ElecGSFTracklnPt = -99.;
  if ( theGsfEle.gsfTrack().isNonnull() ) {
    ElecChi2NormGSF = (theGsfEle).gsfTrack()->normalizedChi2();
    ElecGSFNumHits = (theGsfEle).gsfTrack()->numberOfValidHits();
    if ( theGsfEle.gsfTrack()->pt() > 0. ) {
      ElecGSFTrackResol = theGsfEle.gsfTrack()->ptError()/theGsfEle.gsfTrack()->pt();
      ElecGSFTracklnPt = log(theGsfEle.gsfTrack()->pt())*M_LN10;
    }
  }

  //Variables related to the CtfTrack
  Float_t ElecChi2NormKF = -99.;
  Float_t ElecKFNumHits = -99.;
  if ( theGsfEle.closestCtfTrackRef().isNonnull() ) {
    ElecChi2NormKF = (theGsfEle).closestCtfTrackRef()->normalizedChi2();
    ElecKFNumHits = (theGsfEle).closestCtfTrackRef()->numberOfValidHits();
  }

  return MVAValue(TauPt,
                  TauEtaAtEcalEntrance,
                  TauPhi,
                  TauLeadChargedPFCandPt,
                  TauLeadChargedPFCandEtaAtEcalEntrance,
                  TauEmFraction,
                  TauLeadPFChargedHadrHoP,
                  TauLeadPFChargedHadrEoP,
                  TauVisMassIn,
                  TaudCrackEta,
                  TaudCrackPhi,
                  TauHasGsf,
                  TauSignalPFGammaCandsIn,
                  TauSignalPFGammaCandsOut,
                  GammasdEtaInSigCone,
                  GammasdPhiInSigCone,
                  GammasPtInSigCone,
                  GammasdEtaOutSigCone,
                  GammasdPhiOutSigCone,
                  GammasPtOutSigCone,
                  ElecEta,
                  ElecPhi,
                  ElecEtotOverPin,
                  ElecChi2NormGSF,
                  ElecChi2NormKF,
                  ElecGSFNumHits,
                  ElecKFNumHits,
                  ElecGSFTrackResol,
                  ElecGSFTracklnPt,
                  ElecPin,
                  ElecPout,
                  ElecEecal,
                  ElecDeltaEta,
                  ElecDeltaPhi,
                  ElecMvaInSigmaEtaEta,
                  ElecMvaInHadEnergy,
                  ElecMvaInDeltaEta);
}

double AntiElectronIDMVA6::MVAValue(const reco::PFTau& thePFTau)
{
  // === tau variables ===
  float TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTau.signalPFCands();
  for ( const auto & pfCandidate : signalPFCands ) {
    sumEtaTimesEnergy += pfCandidate->positionAtECALEntrance().eta()*pfCandidate->energy();
    sumEnergy += pfCandidate->energy();
  }
  if ( sumEnergy > 0. ) {
    TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  }
  
  float TauLeadChargedPFCandEtaAtEcalEntrance = -99.;
  float TauLeadChargedPFCandPt = -99.;
  for ( const auto & pfCandidate : signalPFCands ) {
    const reco::Track* track = nullptr;
    if ( pfCandidate->trackRef().isNonnull() ) track = pfCandidate->trackRef().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->innerTrack().isNonnull()  ) track = pfCandidate->muonRef()->innerTrack().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->globalTrack().isNonnull() ) track = pfCandidate->muonRef()->globalTrack().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->outerTrack().isNonnull()  ) track = pfCandidate->muonRef()->outerTrack().get();
    else if ( pfCandidate->gsfTrackRef().isNonnull() ) track = pfCandidate->gsfTrackRef().get();
    if ( track ) {
      if ( track->pt() > TauLeadChargedPFCandPt ) {
	TauLeadChargedPFCandEtaAtEcalEntrance = pfCandidate->positionAtECALEntrance().eta();
	TauLeadChargedPFCandPt = track->pt();
      }
    }
  }

  Float_t TauPt = thePFTau.pt();
  Float_t TauEmFraction = std::max(thePFTau.emFraction(), (Float_t)0.);
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }

  std::vector<Float_t> GammasdEtaInSigCone;
  std::vector<Float_t> GammasdPhiInSigCone;
  std::vector<Float_t> GammasPtInSigCone;
  std::vector<Float_t> GammasdEtaOutSigCone;
  std::vector<Float_t> GammasdPhiOutSigCone;
  std::vector<Float_t> GammasPtOutSigCone;
  reco::Candidate::LorentzVector pfGammaSum(0,0,0,0);
  reco::Candidate::LorentzVector pfChargedSum(0,0,0,0);
  
  for ( const auto & gamma : thePFTau.signalPFGammaCands() ) {
    float dR = deltaR(gamma->p4(), thePFTau.leadPFChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, thePFTau.pt())));

    // pfGammas inside the tau signal cone
    if (dR < signalrad) {
      if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
        GammasdEtaInSigCone.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
      }
      else {
        GammasdEtaInSigCone.push_back(gamma->eta() - thePFTau.eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - thePFTau.phi());
      }
      GammasPtInSigCone.push_back(gamma->pt());
      pfGammaSum += gamma->p4();
    }
    // pfGammas outside the tau signal cone
    else {
      if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
        GammasdEtaOutSigCone.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
      } 
      else {
        GammasdEtaOutSigCone.push_back(gamma->eta() - thePFTau.eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - thePFTau.phi());
      }
      GammasPtOutSigCone.push_back(gamma->pt());
    }
  }
  
  for ( const auto & charged : thePFTau.signalPFChargedHadrCands() ) {
    float dR = deltaR(charged->p4(), thePFTau.leadPFChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, thePFTau.pt())));
  
    // charged particles inside the tau signal cone
    if (dR < signalrad) {
        pfChargedSum += charged->p4();
    }
  }
  
  Int_t TauSignalPFGammaCandsIn = GammasPtInSigCone.size();
  Int_t TauSignalPFGammaCandsOut = GammasPtOutSigCone.size();
  Float_t TauVisMassIn = (pfGammaSum + pfChargedSum).mass();

  Float_t TauPhi = thePFTau.phi();
  float sumPhiTimesEnergy = 0.;
  float sumEnergyPhi = 0.;
  if ( !usePhiAtEcalEntranceExtrapolation_ ){
    for ( const auto & pfCandidate : signalPFCands ) {
      sumPhiTimesEnergy += pfCandidate->positionAtECALEntrance().phi()*pfCandidate->energy();
      sumEnergyPhi += pfCandidate->energy();
    }
  }
  else{
    TauPhi= -99.;
    for ( const auto & signalPFCand : signalPFCands ) {
      reco::Candidate const*  signalCand = signalPFCand.get();
      float phi = thePFTau.phi();
      math::XYZPoint aPos;
      if ( atECalEntrance(signalCand, aPos) == true ) phi = aPos.Phi();
      sumPhiTimesEnergy += phi*signalCand->energy();     
      sumEnergy += signalCand->energy();
    }
  }
  if ( sumEnergyPhi > 0. ) {
    TauPhi = sumPhiTimesEnergy/sumEnergyPhi;
  }
  Float_t TaudCrackPhi = dCrackPhi(TauPhi, TauEtaAtEcalEntrance);
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance);
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();

  
  // === electron variables ===
  Float_t dummyElecEta = 9.9;

  return MVAValue(TauPt,
                  TauEtaAtEcalEntrance,
                  TauPhi,
                  TauLeadChargedPFCandPt,
                  TauLeadChargedPFCandEtaAtEcalEntrance,
                  TauEmFraction,
                  TauLeadPFChargedHadrHoP,
                  TauLeadPFChargedHadrEoP,
                  TauVisMassIn,
                  TaudCrackEta,
                  TaudCrackPhi,
                  TauHasGsf,
                  TauSignalPFGammaCandsIn,
                  TauSignalPFGammaCandsOut,
                  GammasdEtaInSigCone,
                  GammasdPhiInSigCone,
                  GammasPtInSigCone,
                  GammasdEtaOutSigCone,
                  GammasdPhiOutSigCone,
                  GammasPtOutSigCone,
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
                  0.,
                  0.,
                  0.,
                  0.,
                  0.);
}

double AntiElectronIDMVA6::MVAValue(const pat::Tau& theTau, const pat::Electron& theEle)
{
  // === tau variables ===
  float TauEtaAtEcalEntrance = theTau.etaAtEcalEntrance();
  
  float TauLeadChargedPFCandEtaAtEcalEntrance = theTau.etaAtEcalEntranceLeadChargedCand();
  float TauLeadChargedPFCandPt = theTau.ptLeadChargedCand();

  Float_t TauPt = theTau.pt();
  //Float_t TauEmFraction = std::max(theTau.ecalEnergy()/(theTau.ecalEnergy()+theTau.hcalEnergy()), (Float_t)0.);
  Float_t TauEmFraction = std::max(theTau.emFraction_MVA(), (Float_t)0.);
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( theTau.leadChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = theTau.hcalEnergyLeadChargedHadrCand()/theTau.leadChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = theTau.ecalEnergyLeadChargedHadrCand()/theTau.leadChargedHadrCand()->p();
  }

  std::vector<Float_t> GammasdEtaInSigCone;
  std::vector<Float_t> GammasdPhiInSigCone;
  std::vector<Float_t> GammasPtInSigCone;
  std::vector<Float_t> GammasdEtaOutSigCone;
  std::vector<Float_t> GammasdPhiOutSigCone;
  std::vector<Float_t> GammasPtOutSigCone;
  reco::Candidate::LorentzVector pfGammaSum(0,0,0,0);
  reco::Candidate::LorentzVector pfChargedSum(0,0,0,0);
  
  const reco::CandidatePtrVector signalGammaCands = theTau.signalGammaCands();
  for ( const auto & gamma : signalGammaCands ) {
    float dR = deltaR(gamma->p4(), theTau.leadChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, theTau.pt())));

    // pfGammas inside the tau signal cone
    if (dR < signalrad) {
      if ( theTau.leadChargedHadrCand().isNonnull() ) {
        GammasdEtaInSigCone.push_back(gamma->eta() - theTau.leadChargedHadrCand()->eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - theTau.leadChargedHadrCand()->phi());
	//A.-C. please check whether this change is safe against future trainings
        //GammasdPhiInSigCone.push_back(deltaPhi((*gamma)->phi(), theTau.leadChargedHadrCand()->phi()));
      }
      else {
        GammasdEtaInSigCone.push_back(gamma->eta() - theTau.eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - theTau.phi());
	//A.-C. please check whether this change is safe against future trainings	
        //GammasdPhiInSigCone.push_back(deltaPhi(gamma->phi(), theTau.phi()));
      }
      GammasPtInSigCone.push_back(gamma->pt());
      pfGammaSum += gamma->p4();
    }
    // pfGammas outside the tau signal cone
    else {
      if ( theTau.leadChargedHadrCand().isNonnull() ) {
        GammasdEtaOutSigCone.push_back(gamma->eta() - theTau.leadChargedHadrCand()->eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - theTau.leadChargedHadrCand()->phi());
	//A.-C. please check whether this change is safe against future trainings		
        //GammasdPhiOutSigCone.push_back(deltaPhi(gamma->phi(), theTau.leadChargedHadrCand()->phi()));
      } 
      else {
        GammasdEtaOutSigCone.push_back(gamma->eta() - theTau.eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - theTau.phi());
	//A.-C. please chaekc whether this change is safe against future trainings		
        //GammasdPhiOutSigCone.push_back(deltaPhi(gamma->phi(), theTau.phi()));
      }
      GammasPtOutSigCone.push_back(gamma->pt());
    }
  }
  
  const reco::CandidatePtrVector signalChargedCands = theTau.signalChargedHadrCands();
  for ( const auto & charged : signalChargedCands ) {
    float dR = deltaR(charged->p4(), theTau.leadChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, theTau.pt())));
  
    // charged particles inside the tau signal cone
    if (dR < signalrad) {
      pfChargedSum += charged->p4();
    }
  }
  
  Int_t TauSignalPFGammaCandsIn = GammasPtInSigCone.size();
  Int_t TauSignalPFGammaCandsOut = GammasPtOutSigCone.size();
  Float_t TauVisMassIn = (pfGammaSum + pfChargedSum).mass();
  Float_t TauPhi = -99.;
  if ( usePhiAtEcalEntranceExtrapolation_ ) {
    float sumPhiTimesEnergy = 0.;
    float sumEnergy = 0.;
    const reco::CandidatePtrVector signalCands = theTau.signalCands();
    for ( const auto & signalCandPtr : signalCands ) {
      reco::Candidate const* signalCand = signalCandPtr.get();
      float phi = theTau.phi();
      math::XYZPoint aPos;
      if ( atECalEntrance(signalCand, aPos) == true ) phi = aPos.Phi();
      sumPhiTimesEnergy += phi*signalCand->energy();	  
      sumEnergy += signalCand->energy();
    }
    if ( sumEnergy > 0. ) {
      TauPhi = sumPhiTimesEnergy/sumEnergy;
    }
  }
  else {
    TauPhi = theTau.phiAtEcalEntrance();
  } 

  Float_t TaudCrackPhi = dCrackPhi(TauPhi, TauEtaAtEcalEntrance);
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance); 
  
  Float_t TauHasGsf = 0;
  pat::PackedCandidate const* packedLeadTauCand = dynamic_cast<pat::PackedCandidate const*>(theTau.leadChargedHadrCand().get());
  if( abs(packedLeadTauCand->pdgId()) == 11 ) TauHasGsf = 1;
  
  // === electron variables ===
  Float_t ElecEta = theEle.eta();
  Float_t ElecPhi = theEle.phi();
                  
  //Variables related to the electron Cluster
  Float_t ElecEe = 0.;
  Float_t ElecEgamma = 0.;
  reco::SuperClusterRef pfSuperCluster = theEle.superCluster();
  if ( pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable() ) {
    for ( reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin(); pfCluster != pfSuperCluster->clustersEnd(); ++pfCluster ) {
      double pfClusterEn = (*pfCluster)->energy();
      if ( pfCluster == pfSuperCluster->clustersBegin() ) ElecEe += pfClusterEn;
      else ElecEgamma += pfClusterEn;
    }
  }
  
  Float_t ElecPin = std::sqrt(theEle.trackMomentumAtVtx().Mag2());
  Float_t ElecPout = std::sqrt(theEle.trackMomentumOut().Mag2());  
  Float_t ElecEtotOverPin = (ElecPin > 0.0) ? ((ElecEe + ElecEgamma)/ElecPin) : -0.1;
  Float_t ElecEecal = theEle.ecalEnergy();
  Float_t ElecDeltaEta = theEle.deltaEtaSeedClusterTrackAtCalo();
  Float_t ElecDeltaPhi = theEle.deltaPhiSeedClusterTrackAtCalo();
  Float_t ElecMvaInSigmaEtaEta = (theEle).mvaInput().sigmaEtaEta;
  Float_t ElecMvaInHadEnergy = (theEle).mvaInput().hadEnergy;
  Float_t ElecMvaInDeltaEta = (theEle).mvaInput().deltaEta;
  
  //Variables related to the GsfTrack
  Float_t ElecChi2NormGSF = -99.;
  Float_t ElecGSFNumHits = -99.;
  Float_t ElecGSFTrackResol = -99.;
  Float_t ElecGSFTracklnPt = -99.;
  if ( theEle.gsfTrack().isNonnull() ) {
    ElecChi2NormGSF = (theEle).gsfTrack()->normalizedChi2();
    ElecGSFNumHits = (theEle).gsfTrack()->numberOfValidHits();
    if ( theEle.gsfTrack()->pt() > 0. ) {
      ElecGSFTrackResol = theEle.gsfTrack()->ptError()/theEle.gsfTrack()->pt();
      ElecGSFTracklnPt = log(theEle.gsfTrack()->pt())*M_LN10;
    }
  }

  //Variables related to the CtfTrack
  Float_t ElecChi2NormKF = -99.;
  Float_t ElecKFNumHits = -99.;
  if ( theEle.closestCtfTrackRef().isNonnull() ) {
    ElecChi2NormKF = (theEle).closestCtfTrackRef()->normalizedChi2();
    ElecKFNumHits = (theEle).closestCtfTrackRef()->numberOfValidHits();
  }

  return MVAValue(TauPt,
                  TauEtaAtEcalEntrance,
                  TauPhi,
                  TauLeadChargedPFCandPt,
                  TauLeadChargedPFCandEtaAtEcalEntrance,
                  TauEmFraction,
                  TauLeadPFChargedHadrHoP,
                  TauLeadPFChargedHadrEoP,
                  TauVisMassIn,
                  TaudCrackEta,
                  TaudCrackPhi,
                  TauHasGsf,
                  TauSignalPFGammaCandsIn,
                  TauSignalPFGammaCandsOut,
                  GammasdEtaInSigCone,
                  GammasdPhiInSigCone,
                  GammasPtInSigCone,
                  GammasdEtaOutSigCone,
                  GammasdPhiOutSigCone,
                  GammasPtOutSigCone,
                  ElecEta,
                  ElecPhi,
                  ElecEtotOverPin,
                  ElecChi2NormGSF,
                  ElecChi2NormKF,
                  ElecGSFNumHits,
                  ElecKFNumHits,
                  ElecGSFTrackResol,
                  ElecGSFTracklnPt,
                  ElecPin,
                  ElecPout,
                  ElecEecal,
                  ElecDeltaEta,
                  ElecDeltaPhi,
                  ElecMvaInSigmaEtaEta,
                  ElecMvaInHadEnergy,
                  ElecMvaInDeltaEta);
}

double AntiElectronIDMVA6::MVAValue(const pat::Tau& theTau)
{
  // === tau variables ===
  float TauEtaAtEcalEntrance = theTau.etaAtEcalEntrance();
  
  float TauLeadChargedPFCandEtaAtEcalEntrance = theTau.etaAtEcalEntranceLeadChargedCand();
  float TauLeadChargedPFCandPt = theTau.ptLeadChargedCand();

  Float_t TauPt = theTau.pt();
  //Float_t TauEmFraction = std::max(theTau.ecalEnergy()/(theTau.ecalEnergy()+theTau.hcalEnergy()), (Float_t)0.);
  Float_t TauEmFraction = std::max(theTau.emFraction_MVA(), (Float_t)0.);
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( theTau.leadChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = theTau.hcalEnergyLeadChargedHadrCand()/theTau.leadChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = theTau.ecalEnergyLeadChargedHadrCand()/theTau.leadChargedHadrCand()->p();
  }

  std::vector<Float_t> GammasdEtaInSigCone;
  std::vector<Float_t> GammasdPhiInSigCone;
  std::vector<Float_t> GammasPtInSigCone;
  std::vector<Float_t> GammasdEtaOutSigCone;
  std::vector<Float_t> GammasdPhiOutSigCone;
  std::vector<Float_t> GammasPtOutSigCone;
  reco::Candidate::LorentzVector pfGammaSum(0,0,0,0);
  reco::Candidate::LorentzVector pfChargedSum(0,0,0,0);
  
  const reco::CandidatePtrVector signalGammaCands = theTau.signalGammaCands();
  for ( const auto & gamma : signalGammaCands ) {
    float dR = deltaR(gamma->p4(), theTau.leadChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, theTau.pt())));

    // pfGammas inside the tau signal cone
    if (dR < signalrad) {
      if ( theTau.leadChargedHadrCand().isNonnull() ) {
        GammasdEtaInSigCone.push_back(gamma->eta() - theTau.leadChargedHadrCand()->eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - theTau.leadChargedHadrCand()->phi());
      }
      else {
        GammasdEtaInSigCone.push_back(gamma->eta() - theTau.eta());
        GammasdPhiInSigCone.push_back(gamma->phi() - theTau.phi());
      }
      GammasPtInSigCone.push_back(gamma->pt());
      pfGammaSum += gamma->p4();
    }
    // pfGammas outside the tau signal cone
    else {
      if ( theTau.leadChargedHadrCand().isNonnull() ) {
        GammasdEtaOutSigCone.push_back(gamma->eta() - theTau.leadChargedHadrCand()->eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - theTau.leadChargedHadrCand()->phi());
      } 
      else {
        GammasdEtaOutSigCone.push_back(gamma->eta() - theTau.eta());
        GammasdPhiOutSigCone.push_back(gamma->phi() - theTau.phi());
      }
      GammasPtOutSigCone.push_back(gamma->pt());
    }
  }
  
  const reco::CandidatePtrVector signalChargedCands = theTau.signalChargedHadrCands();
  for ( const auto & charged : signalChargedCands ) {
    float dR = deltaR(charged->p4(), theTau.leadChargedHadrCand()->p4());
    float signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, theTau.pt())));
  
    // charged particles inside the tau signal cone
    if (dR < signalrad) {
        pfChargedSum += charged->p4();
    }
  }
  
  Int_t TauSignalPFGammaCandsIn = GammasPtInSigCone.size();
  Int_t TauSignalPFGammaCandsOut = GammasPtOutSigCone.size();
  Float_t TauVisMassIn = (pfGammaSum + pfChargedSum).mass();
  Float_t TauPhi = -99.;
  if ( usePhiAtEcalEntranceExtrapolation_ ) {
    float sumPhiTimesEnergy = 0.;
    float sumEnergy = 0.;
    const reco::CandidatePtrVector signalCands = theTau.signalCands();
    for ( const auto & signalCandPtr : signalCands ) {
      reco::Candidate const* signalCand = signalCandPtr.get();
      float phi = theTau.phi();
      math::XYZPoint aPos;
      if ( atECalEntrance(signalCand, aPos) == true ) phi = aPos.Phi();
      sumPhiTimesEnergy += phi*signalCand->energy();	  
      sumEnergy += signalCand->energy();
    }
    if ( sumEnergy > 0. ) {
      TauPhi = sumPhiTimesEnergy/sumEnergy;
    }
  }
  else {
    TauPhi = theTau.phiAtEcalEntrance();
  } 

  Float_t TaudCrackPhi = dCrackPhi(TauPhi, TauEtaAtEcalEntrance);
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance); 
  
  Float_t TauHasGsf = 0;
  pat::PackedCandidate const* packedLeadTauCand = dynamic_cast<pat::PackedCandidate const*>(theTau.leadChargedHadrCand().get());
  //const reco::Track & pseudoTrack = packedLeadTauCand->pseudoTrack();
  if( abs(packedLeadTauCand->pdgId()) == 11 ) TauHasGsf = 1;
  
  // === electron variables ===
  Float_t dummyElecEta = 9.9;

  return MVAValue(TauPt,
                  TauEtaAtEcalEntrance,
                  TauPhi,
                  TauLeadChargedPFCandPt,
                  TauLeadChargedPFCandEtaAtEcalEntrance,
                  TauEmFraction,
                  TauLeadPFChargedHadrHoP,
                  TauLeadPFChargedHadrEoP,
                  TauVisMassIn,
                  TaudCrackEta,
                  TaudCrackPhi,
                  TauHasGsf,
                  TauSignalPFGammaCandsIn,
                  TauSignalPFGammaCandsOut,
                  GammasdEtaInSigCone,
                  GammasdPhiInSigCone,
                  GammasPtInSigCone,
                  GammasdEtaOutSigCone,
                  GammasdPhiOutSigCone,
                  GammasPtOutSigCone,
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
                  0.,
                  0.,
                  0.,
                  0.,
                  0.);
}

double AntiElectronIDMVA6::minimum(double a, double b)
{
  if ( std::abs(b) < std::abs(a) ) return b;
  else return a;
}

namespace {

  // IN: define locations of the 18 phi-cracks
 std::array<double,18> fill_cPhi() {
   constexpr double pi = M_PI; // 3.14159265358979323846;
   std::array<double,18> cPhi;
   // IN: define locations of the 18 phi-cracks
   cPhi[0] = 2.97025;
   for ( unsigned iCrack = 1; iCrack <= 17; ++iCrack ) {
     cPhi[iCrack] = cPhi[0] - 2.*iCrack*pi/18;
   }
   return cPhi;
 }
     
  const std::array<double,18> cPhi = fill_cPhi();

}

double AntiElectronIDMVA6::dCrackPhi(double phi, double eta)
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

double AntiElectronIDMVA6::dCrackEta(double eta)
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

bool AntiElectronIDMVA6::atECalEntrance(const reco::Candidate* part, math::XYZPoint &pos)
{
  bool result = false;
  BaseParticlePropagator theParticle =
    BaseParticlePropagator(RawParticle(math::XYZTLorentzVector(part->px(),
							       part->py(),
							       part->pz(),
							       part->energy()),
				       math::XYZTLorentzVector(part->vertex().x(),
							       part->vertex().y(),
							       part->vertex().z(),
							       0.)), 
			   0.,0.,bField_);
  theParticle.setCharge(part->charge());
  theParticle.propagateToEcalEntrance(false);
  if(theParticle.getSuccess()!=0){
    pos = math::XYZPoint(theParticle.vertex());
    result = true;
  }
  else {
    result = false;
  }
  return result;
}
