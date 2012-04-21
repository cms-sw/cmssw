#include "RecoMET/METAlgorithms/interface/PFMETAlgorithmMVA.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/CommonMETData.h"

#include <TFile.h>
#include <TMath.h>

enum MVAType { kBaseline = 0 };

GBRForest* loadMVA(const edm::FileInPath& inputFileName, const std::string& mvaName)
{
  if ( !inputFileName.isLocal()) throw cms::Exception("PFMETAlgorithmMVA::loadMVA") 
    << " Failed to find File = " << inputFileName << " !!\n";
  TFile* inputFile = new TFile(inputFileName.fullPath().data());

  GBRForest* mva = dynamic_cast<GBRForest*>(inputFile->Get(mvaName.data()));
  if ( !mva )
    throw cms::Exception("PFMETAlgorithmMVA::loadMVA")
      << " Failed to load MVA = " << mvaName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";
  
  delete inputFile;

  return mva;
}

PFMETAlgorithmMVA::PFMETAlgorithmMVA(const edm::ParameterSet& cfg) 
  : utils_(cfg),
    mvaInputU_(0),
    mvaInputDPhi_(0),
    mvaInputCovU1_(0),
    mvaInputCovU2_(0),
    mvaMEtCov_(2, 2),
    mvaReaderU_(0),
    mvaReaderDPhi_(0),
    mvaReaderCovU1_(0),
    mvaReaderCovU2_(0)
{
  mvaType_ = kBaseline;

  dZcut_ = cfg.getParameter<double>("dZcut");
  
  mvaNameU_      = "U1Correction";
  mvaNameDPhi_   = "PhiCorrection";
  mvaNameCovU1_  = "CovU1";
  mvaNameCovU2_  = "CovU2";

  edm::ParameterSet cfgInputFileNames = cfg.getParameter<edm::ParameterSet>("inputFileNames");
  edm::FileInPath inputFileNameU = cfgInputFileNames.getParameter<edm::FileInPath>("U");
  mvaReaderU_ = loadMVA(inputFileNameU, mvaNameU_);
  edm::FileInPath inputFileNameDPhi = cfgInputFileNames.getParameter<edm::FileInPath>("DPhi");
  mvaReaderDPhi_ = loadMVA(inputFileNameDPhi, mvaNameDPhi_);
  edm::FileInPath inputFileNameCovU1 = cfgInputFileNames.getParameter<edm::FileInPath>("CovU1");
  mvaReaderCovU1_ = loadMVA(inputFileNameCovU1, mvaNameCovU1_);
  edm::FileInPath inputFileNameCovU2 = cfgInputFileNames.getParameter<edm::FileInPath>("CovU2");
  mvaReaderCovU2_ = loadMVA(inputFileNameCovU2, mvaNameCovU2_);

  mvaInputU_     = new Float_t[25];
  mvaInputDPhi_  = new Float_t[23];
  mvaInputCovU1_ = new Float_t[26];
  mvaInputCovU2_ = new Float_t[26];
}

PFMETAlgorithmMVA::~PFMETAlgorithmMVA()
{
  delete mvaInputU_;
  delete mvaInputDPhi_;
  delete mvaInputCovU1_;
  delete mvaInputCovU2_;

  delete mvaReaderU_;
  delete mvaReaderDPhi_;
  delete mvaReaderCovU1_;
  delete mvaReaderCovU2_;
}

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::setInput(const std::vector<reco::Candidate::LorentzVector>& leptons,
				 const std::vector<mvaMEtUtilities::JetInfo>& jets,
				 const std::vector<mvaMEtUtilities::pfCandInfo>& pfCandidates,
				 const std::vector<reco::Vertex::Point>& vertices)
{
  CommonMETData sumLeptons;
  sumLeptons.mex   = 0.;
  sumLeptons.mey   = 0.;
  sumLeptons.sumet = 0.;
  for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	lepton != leptons.end(); ++lepton ) {
    sumLeptons.mex   += lepton->px();
    sumLeptons.mey   += lepton->py();
    sumLeptons.sumet += lepton->pt();
  }

  sumLeptonPx_ = sumLeptons.mex;
  sumLeptonPy_ = sumLeptons.mey;

  std::vector<mvaMEtUtilities::JetInfo> jets_cleaned = utils_.cleanJets(jets, leptons);

  CommonMETData pfRecoil_data  = utils_.computePFRecoil(sumLeptons, pfCandidates, dZcut_);
  CommonMETData tkRecoil_data  = utils_.computeTrackRecoil(sumLeptons, pfCandidates, dZcut_);
  CommonMETData npuRecoil_data = utils_.computeNoPURecoil(sumLeptons, pfCandidates, jets_cleaned, dZcut_);
  CommonMETData pucRecoil_data = utils_.computePUCRecoil(sumLeptons, pfCandidates, jets_cleaned, dZcut_);
  CommonMETData puMEt_data     = utils_.computePUMEt(pfCandidates, jets_cleaned, dZcut_);

  reco::Candidate::LorentzVector jet1P4 = utils_.leadJetP4(jets);
  reco::Candidate::LorentzVector jet2P4 = utils_.subleadJetP4(jets);

  double pfSumEt       = pfRecoil_data.sumet;
  double pfU           = pfRecoil_data.met;
  double pfPhi         = pfRecoil_data.phi;
  double tkSumEt       = tkRecoil_data.sumet;
  double tkU           = tkRecoil_data.met;
  double tkPhi         = tkRecoil_data.phi;
  double npuSumEt      = npuRecoil_data.sumet;
  double npuU          = npuRecoil_data.met; 
  double npuPhi        = npuRecoil_data.phi;
  double puSumEt       = puMEt_data.sumet;
  double puMEt         = puMEt_data.met;
  double puPhi         = puMEt_data.phi;
  double pucSumEt      = pucRecoil_data.sumet; 
  double pucU          = pucRecoil_data.met; 
  double pucPhi        = pucRecoil_data.phi;
  double jet1Pt        = jet1P4.pt();
  double jet1Eta       = jet1P4.eta();
  double jet1Phi       = jet1P4.phi();
  double jet2Pt        = jet2P4.pt();
  double jet2Eta       = jet2P4.eta();
  double jet2Phi       = jet2P4.phi();
  double numJetsPtGt30 = utils_.numJetsAboveThreshold(jets, 30.);
  double numJets       = jets.size();
  double numVertices   = vertices.size();

  setInput(pfSumEt, pfU, pfPhi,
	   tkSumEt, tkU, tkPhi,
	   npuSumEt, npuU, npuPhi,
	   puSumEt, puMEt, puPhi,
	   pucSumEt, pucU, pucPhi,
	   jet1Pt, jet1Eta, jet1Phi,
	   jet2Pt, jet2Eta, jet2Phi,
	   numJetsPtGt30, numJets, 
	   numVertices);
}

void PFMETAlgorithmMVA::setInput(double pfSumEt, double pfU, double pfPhi,
				 double tkSumEt, double tkU, double tkPhi,
				 double npuSumEt, double npuU, double npuPhi,
				 double puSumEt, double puMEt, double puPhi,
				 double pucSumEt, double pucU, double pucPhi,
				 double jet1Pt, double jet1Eta, double jet1Phi,
				 double jet2Pt, double jet2Eta, double jet2Phi,
				 double numJetsPtGt30, double numJets, 
				 double numVertices)
{
  // CV: add protection against "empty events"
  if ( pfSumEt < 1. ) pfSumEt = 1.;
  
  pfSumEt_       = pfSumEt;
  pfU_           = pfU;
  pfPhi_         = pfPhi;
  tkSumEt_       = tkSumEt/pfSumEt_;
  tkU_           = tkU;
  tkPhi_         = tkPhi;
  npuSumEt_      = npuSumEt/pfSumEt_;
  npuU_          = npuU;
  npuPhi_        = npuPhi;
  puSumEt_       = puSumEt/pfSumEt_;
  puMEt_         = puMEt;
  puPhi_         = puPhi;
  pucSumEt_      = pucSumEt/pfSumEt_;
  pucU_          = pucU;
  pucPhi_        = pucPhi;
  jet1Pt_        = jet1Pt;
  jet1Eta_       = jet1Eta;
  jet1Phi_       = jet1Phi;
  jet2Pt_        = jet2Pt;
  jet2Eta_       = jet2Eta;
  jet2Phi_       = jet2Phi;
  numJetsPtGt30_ = numJetsPtGt30;
  numJets_       = numJets;
  numVertices_   = numVertices;
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::evaluateMVA()
{
  // CV: MVAs needs to be evaluated in order { DPhi, U1, CovU1, CovU2 }
  //     as MVA for U1 (CovU1, CovU2) uses output of DPhi (DPhi and U1) MVA
  evaluateDPhi();
  evaluateU();
  evaluateCovU1();
  evaluateCovU2();

  // compute MET
  double U      = pfU_*mvaOutputU_;
  double Phi    = pfPhi_ + mvaOutputDPhi_;
  if ( U < 0. ) Phi += TMath::Pi();
  double cosPhi = cos(Phi);
  double sinPhi = sin(Phi);
  double metPx  = U*cosPhi - sumLeptonPx_; // CV: U is actually minus the hadronic recoil in the event
  double metPy  = U*sinPhi - sumLeptonPy_;
  double metPt  = sqrt(metPx*metPx + metPy*metPy);
  mvaMEt_.SetCoordinates(metPx, metPy, 0., metPt);

  // compute MET uncertainties in dirrections parallel and perpendicular to hadronic recoil
  // (neglecting uncertainties on lepton momenta)
  mvaMEtCov_(0, 0) =  mvaOutputCovU1_*cosPhi*cosPhi + mvaOutputCovU2_*sinPhi*sinPhi;
  mvaMEtCov_(0, 1) = -mvaOutputCovU1_*sinPhi*cosPhi + mvaOutputCovU2_*sinPhi*cosPhi;
  mvaMEtCov_(1, 0) = mvaMEtCov_(0, 1);
  mvaMEtCov_(1, 1) =  mvaOutputCovU1_*sinPhi*sinPhi + mvaOutputCovU2_*cosPhi*cosPhi;
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::evaluateU() 
{ 
  mvaInputU_[0]  = pfSumEt_; // PH: helps flattens response vs. Nvtx
  mvaInputU_[1]  = numVertices_;
  mvaInputU_[2]  = pfU_;
  mvaInputU_[3]  = pfPhi_;
  mvaInputU_[4]  = tkSumEt_;
  mvaInputU_[5]  = tkU_;
  mvaInputU_[6]  = tkPhi_;
  mvaInputU_[7]  = npuSumEt_;
  mvaInputU_[8]  = npuU_;
  mvaInputU_[9]  = npuPhi_;
  mvaInputU_[10] = puSumEt_;
  mvaInputU_[11] = puMEt_;
  mvaInputU_[12] = puPhi_;
  mvaInputU_[13] = pucSumEt_;
  mvaInputU_[14] = pucU_;
  mvaInputU_[15] = pucPhi_;
  mvaInputU_[16] = jet1Pt_;
  mvaInputU_[17] = jet1Eta_;
  mvaInputU_[18] = jet1Phi_;
  mvaInputU_[19] = jet2Pt_;
  mvaInputU_[20] = jet2Eta_;
  mvaInputU_[21] = jet2Phi_;
  mvaInputU_[22] = numJets_;
  mvaInputU_[23] = numJetsPtGt30_;
  mvaInputU_[24] = pfPhi_ + mvaOutputDPhi_;
  mvaOutputU_    = mvaReaderU_->GetResponse(mvaInputU_);
}

void PFMETAlgorithmMVA::evaluateDPhi() 
{ 
  mvaInputDPhi_[0]  = numVertices_;
  mvaInputDPhi_[1]  = pfU_;
  mvaInputDPhi_[2]  = pfPhi_;
  mvaInputDPhi_[3]  = tkSumEt_;
  mvaInputDPhi_[4]  = tkU_;
  mvaInputDPhi_[5]  = tkPhi_;
  mvaInputDPhi_[6]  = npuSumEt_;
  mvaInputDPhi_[7]  = npuU_;
  mvaInputDPhi_[8]  = npuPhi_;
  mvaInputDPhi_[9]  = puSumEt_;
  mvaInputDPhi_[10] = puMEt_;
  mvaInputDPhi_[11] = puPhi_;
  mvaInputDPhi_[12] = pucSumEt_;
  mvaInputDPhi_[13] = pucU_;
  mvaInputDPhi_[14] = pucPhi_;
  mvaInputDPhi_[15] = jet1Pt_;
  mvaInputDPhi_[16] = jet1Eta_;
  mvaInputDPhi_[17] = jet1Phi_;
  mvaInputDPhi_[18] = jet2Pt_;
  mvaInputDPhi_[19] = jet2Eta_;
  mvaInputDPhi_[20] = jet2Phi_;
  mvaInputDPhi_[21] = numJets_;
  mvaInputDPhi_[22] = numJetsPtGt30_;
  mvaOutputDPhi_    = mvaReaderDPhi_->GetResponse(mvaInputDPhi_);
}

void PFMETAlgorithmMVA::evaluateCovU1() 
{ 
  mvaInputCovU1_[0]  = pfSumEt_; // PH: helps flattens response vs. Nvtx
  mvaInputCovU1_[1]  = numVertices_;
  mvaInputCovU1_[2]  = pfU_;
  mvaInputCovU1_[3]  = pfPhi_;
  mvaInputCovU1_[4]  = tkSumEt_;
  mvaInputCovU1_[5]  = tkU_;
  mvaInputCovU1_[6]  = tkPhi_;
  mvaInputCovU1_[7]  = npuSumEt_;
  mvaInputCovU1_[8]  = npuU_;
  mvaInputCovU1_[9]  = npuPhi_;
  mvaInputCovU1_[10] = puSumEt_;
  mvaInputCovU1_[11] = puMEt_;
  mvaInputCovU1_[12] = puPhi_;
  mvaInputCovU1_[13] = pucSumEt_;
  mvaInputCovU1_[14] = pucU_;
  mvaInputCovU1_[15] = pucPhi_;
  mvaInputCovU1_[16] = jet1Pt_;
  mvaInputCovU1_[17] = jet1Eta_;
  mvaInputCovU1_[18] = jet1Phi_;
  mvaInputCovU1_[19] = jet2Pt_;
  mvaInputCovU1_[20] = jet2Eta_;
  mvaInputCovU1_[21] = jet2Phi_;
  mvaInputCovU1_[22] = numJets_;
  mvaInputCovU1_[23] = numJetsPtGt30_;
  mvaInputCovU1_[24] = pfPhi_ + mvaOutputDPhi_;
  mvaInputCovU1_[25] = mvaOutputU_;
  mvaOutputCovU1_    = mvaReaderCovU1_->GetResponse(mvaInputCovU1_);
}

void PFMETAlgorithmMVA::evaluateCovU2() 
{ 
  mvaInputCovU2_[0]  = pfSumEt_; // PH: helps flattens response vs. Nvtx
  mvaInputCovU2_[1]  = numVertices_;
  mvaInputCovU2_[2]  = pfU_;
  mvaInputCovU2_[3]  = pfPhi_;
  mvaInputCovU2_[4]  = tkSumEt_;
  mvaInputCovU2_[5]  = tkU_;
  mvaInputCovU2_[6]  = tkPhi_;
  mvaInputCovU2_[7]  = npuSumEt_;
  mvaInputCovU2_[8]  = npuU_;
  mvaInputCovU2_[9]  = npuPhi_;
  mvaInputCovU2_[10] = puSumEt_;
  mvaInputCovU2_[11] = puMEt_;
  mvaInputCovU2_[12] = puPhi_;
  mvaInputCovU2_[13] = pucSumEt_;
  mvaInputCovU2_[14] = pucU_;
  mvaInputCovU2_[15] = pucPhi_;
  mvaInputCovU2_[16] = jet1Pt_;
  mvaInputCovU2_[17] = jet1Eta_;
  mvaInputCovU2_[18] = jet1Phi_;
  mvaInputCovU2_[19] = jet2Pt_;
  mvaInputCovU2_[20] = jet2Eta_;
  mvaInputCovU2_[21] = jet2Phi_;
  mvaInputCovU2_[22] = numJets_;
  mvaInputCovU2_[23] = numJetsPtGt30_;
  mvaInputCovU2_[24] = pfPhi_ + mvaOutputDPhi_;
  mvaInputCovU2_[25] = mvaOutputU_;
  mvaOutputCovU2_    = mvaReaderCovU2_->GetResponse(mvaInputCovU2_);
}
//-------------------------------------------------------------------------------
