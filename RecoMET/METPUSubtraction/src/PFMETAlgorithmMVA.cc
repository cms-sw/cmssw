#include "RecoMET/METPUSubtraction/interface/PFMETAlgorithmMVA.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include "DataFormats/METReco/interface/CommonMETData.h"

#include <TFile.h>

#include <iomanip>

enum MVAType { kBaseline = 0 };

const double Pi=std::cos(-1);

const std::string PFMETAlgorithmMVA::updateVariableNames(std::string input)
{
  if(input=="sumet")     return "particleFlow_SumET";
  if(input=="npv")       return "nPV";
  if(input=="pfu")       return "particleFlow_U";
  if(input=="pfuphi")    return "particleFlow_UPhi";
  if(input=="tksumet")   return "track_SumET";
  if(input=="tku")       return "track_U";
  if(input=="tkuphi")    return "track_UPhi";
  if(input=="nopusumet") return "noPileUp_SumET";
  if(input=="nopuu")     return "noPileUp_U";
  if(input=="nopuuphi")  return "noPileUp_UPhi";
  if(input=="pusumet")   return "pileUp_SumET";
  if(input=="pumet")     return "pileUp_MET";
  if(input=="pumetphi")  return "pileUp_METPhi";
  if(input=="pucsumet")  return "pileUpCorrected_SumET";
  if(input=="pucu")      return "pileUpCorrected_U";
  if(input=="pucuphi")   return "pileUpCorrected_UPhi";
  if(input=="jetpt1")    return "jet1_pT";
  if(input=="jeteta1")   return "jet1_eta";
  if(input=="jetphi1")   return "jet1_Phi";
  if(input=="jetpt2")    return "jet2_pT";
  if(input=="jeteta2")   return "jet2_eta";
  if(input=="jetphi2")   return "jet2_Phi";
  if(input=="nalljet")   return "nJets";
  if(input=="njet")      return "numJetsPtGt30";
  if(input=="uphi_mva")  return "PhiCor_UPhi";
  if(input=="uphix_mva") return "PhiCor_UPhi";
  if(input=="ux_mva")    return "RecoilCor_U";
  return input;
}

const GBRForest* PFMETAlgorithmMVA::loadMVAfromFile(const edm::FileInPath& inputFileName, const std::string& mvaName)
{
  if ( inputFileName.location()==edm::FileInPath::Unknown ) throw cms::Exception("PFMETAlgorithmMVA::loadMVA") 
    << " Failed to find File = " << inputFileName << " !!\n";
  std::unique_ptr<TFile> inputFile(new TFile(inputFileName.fullPath().data()) );

  std::vector<std::string> *lVec = (std::vector<std::string>*)inputFile->Get("varlist");

  if(lVec==nullptr) {
    throw cms::Exception("PFMETAlgorithmMVA::loadMVA")
      << " Failed to load mva file : " << inputFileName.fullPath().data() << " is not a proper file !!\n";
  }

  std::vector<std::string> variableNames;
  for(unsigned int i=0; i< lVec->size();++i)
  {
      variableNames.push_back(updateVariableNames(lVec->at(i)));
  }

  if(mvaName.find(mvaNameU_)      != std::string::npos) varForU_    = variableNames;
  else if(mvaName.find(mvaNameDPhi_)   != std::string::npos) varForDPhi_ = variableNames;
  else if(mvaName.find(mvaNameCovU1_)  != std::string::npos) varForCovU1_ = variableNames;
  else if(mvaName.find(mvaNameCovU2_)  != std::string::npos) varForCovU2_ = variableNames;
  else throw cms::Exception("PFMETAlgorithmMVA::loadMVA") << "MVA MET weight file tree names do not match specified inputs" << std::endl;


  const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
  if ( !mva )
    throw cms::Exception("PFMETAlgorithmMVA::loadMVA")
      << " Failed to load MVA = " << mvaName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";

  return mva;
}

const GBRForest* PFMETAlgorithmMVA::loadMVAfromDB(const edm::EventSetup& es, const std::string& mvaName)
{
  edm::ESHandle<GBRForest> mva;
  es.get<GBRWrapperRcd>().get(mvaName, mva);
  return mva.product();
}

PFMETAlgorithmMVA::PFMETAlgorithmMVA(const edm::ParameterSet& cfg) 
  : utils_(cfg),
    mvaReaderU_(nullptr),
    mvaReaderDPhi_(nullptr),
    mvaReaderCovU1_(nullptr),
    mvaReaderCovU2_(nullptr),
    cfg_(cfg)
{
  mvaType_ = kBaseline;

  loadMVAfromDB_ = cfg.getParameter<bool>("loadMVAfromDB");
  
}

PFMETAlgorithmMVA::~PFMETAlgorithmMVA()
{
  if ( !loadMVAfromDB_ ) {
    delete mvaReaderU_;
    delete mvaReaderDPhi_;
    delete mvaReaderCovU1_;
    delete mvaReaderCovU2_;
  }
}

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::initialize(const edm::EventSetup& es)
{
  edm::ParameterSet cfgInputRecords = cfg_.getParameter<edm::ParameterSet>("inputRecords");
  mvaNameU_       = cfgInputRecords.getParameter<std::string>("U");
  mvaNameDPhi_    = cfgInputRecords.getParameter<std::string>("DPhi");
  mvaNameCovU1_   = cfgInputRecords.getParameter<std::string>("CovU1");
  mvaNameCovU2_   = cfgInputRecords.getParameter<std::string>("CovU2");

  if ( loadMVAfromDB_ ) {
    mvaReaderU_     = loadMVAfromDB(es, mvaNameU_);
    mvaReaderDPhi_  = loadMVAfromDB(es, mvaNameDPhi_);
    mvaReaderCovU1_ = loadMVAfromDB(es, mvaNameCovU1_);
    mvaReaderCovU2_ = loadMVAfromDB(es, mvaNameCovU2_);
  } else {
    edm::ParameterSet cfgInputFileNames = cfg_.getParameter<edm::ParameterSet>("inputFileNames");
    
    edm::FileInPath inputFileNameU = cfgInputFileNames.getParameter<edm::FileInPath>("U");
    mvaReaderU_     = loadMVAfromFile(inputFileNameU, mvaNameU_);
    edm::FileInPath inputFileNameDPhi = cfgInputFileNames.getParameter<edm::FileInPath>("DPhi");
    mvaReaderDPhi_  = loadMVAfromFile(inputFileNameDPhi, mvaNameDPhi_);
    edm::FileInPath inputFileNameCovU1 = cfgInputFileNames.getParameter<edm::FileInPath>("CovU1");
    mvaReaderCovU1_ = loadMVAfromFile(inputFileNameCovU1, mvaNameCovU1_);
    edm::FileInPath inputFileNameCovU2 = cfgInputFileNames.getParameter<edm::FileInPath>("CovU2");
    mvaReaderCovU2_ = loadMVAfromFile(inputFileNameCovU2, mvaNameCovU2_);
  }
}

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::setInput(const std::vector<reco::PUSubMETCandInfo>& leptons,
				 const std::vector<reco::PUSubMETCandInfo>& jets,
				 const std::vector<reco::PUSubMETCandInfo>& pfCandidates,
				 const std::vector<reco::Vertex::Point>& vertices)
{

  
  utils_.computeAllSums( jets, leptons, pfCandidates);
  
  sumLeptonPx_        = utils_.getLeptonsSumMEX();
  sumLeptonPy_        = utils_.getLeptonsSumMEY();

  chargedSumLeptonPx_ = utils_.getLeptonsChSumMEX();
  chargedSumLeptonPy_ = utils_.getLeptonsChSumMEY();

  const std::vector<reco::PUSubMETCandInfo> jets_cleaned = utils_.getCleanedJets();

  CommonMETData pfRecoil_data  = utils_.computeRecoil( MvaMEtUtilities::kPF );
  CommonMETData chHSRecoil_data  = utils_.computeRecoil( MvaMEtUtilities::kChHS );
  CommonMETData hsRecoil_data = utils_.computeRecoil( MvaMEtUtilities::kHS );
  CommonMETData puRecoil_data = utils_.computeRecoil( MvaMEtUtilities::kPU );
  CommonMETData hsMinusNeutralPUMEt_data = utils_.computeRecoil( MvaMEtUtilities::kHSMinusNeutralPU );

  reco::Candidate::LorentzVector jet1P4 = utils_.leadJetP4(jets_cleaned);
  reco::Candidate::LorentzVector jet2P4 = utils_.subleadJetP4(jets_cleaned);


  var_["particleFlow_U"]        = pfRecoil_data.met;
  var_["particleFlow_SumET"]    = pfRecoil_data.sumet;
  var_["particleFlow_UPhi"]     = pfRecoil_data.phi;

  var_["track_SumET"]           = chHSRecoil_data.sumet/var_["particleFlow_SumET"];
  var_["track_U"]               = chHSRecoil_data.met;
  var_["track_UPhi"]            = chHSRecoil_data.phi;

  var_["noPileUp_SumET"]        = hsRecoil_data.sumet/var_["particleFlow_SumET"];
  var_["noPileUp_U"]            = hsRecoil_data.met;
  var_["noPileUp_UPhi"]         = hsRecoil_data.phi;

  var_["pileUp_SumET"]          = puRecoil_data.sumet/var_["particleFlow_SumET"];
  var_["pileUp_MET"]            = puRecoil_data.met;
  var_["pileUp_METPhi"]         = puRecoil_data.phi;

  var_["pileUpCorrected_SumET"] = hsMinusNeutralPUMEt_data.sumet/var_["particleFlow_SumET"];
  var_["pileUpCorrected_U"]     = hsMinusNeutralPUMEt_data.met;
  var_["pileUpCorrected_UPhi"]  = hsMinusNeutralPUMEt_data.phi;

  var_["jet1_pT"]               = jet1P4.pt();
  var_["jet1_eta"]              = jet1P4.eta();
  var_["jet1_Phi"]              = jet1P4.phi();
  var_["jet2_pT"]               = jet2P4.pt();
  var_["jet2_eta"]              = jet2P4.eta();
  var_["jet2_Phi"]              = jet2P4.phi();

  var_["numJetsPtGt30"]         = utils_.numJetsAboveThreshold(jets_cleaned, 30.);
  var_["nJets"]                 = jets_cleaned.size();
  var_["nPV"]                   = vertices.size();
}

//-------------------------------------------------------------------------------
std::unique_ptr<float[]> PFMETAlgorithmMVA::createFloatVector(std::vector<std::string> variableNames)
{
  std::unique_ptr<float[]> floatVector(new float[variableNames.size()]);
    int i = 0;
    for(auto variableName: variableNames)
    {
        floatVector[i++] = var_[variableName];
    }
    return floatVector;
}

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::evaluateMVA()
{
  // CV: MVAs needs to be evaluated in order { DPhi, U1, CovU1, CovU2 }
  //     as MVA for U1 (CovU1, CovU2) uses output of DPhi (DPhi and U1) MVA
  mvaOutputDPhi_  = GetResponse(mvaReaderDPhi_, varForDPhi_);
  var_["PhiCor_UPhi"] = var_["particleFlow_UPhi"] + mvaOutputDPhi_;
  mvaOutputU_     = GetResponse(mvaReaderU_, varForU_);
  var_["RecoilCor_U"] = var_["particleFlow_U"] * mvaOutputU_;
  var_["RecoilCor_UPhi"] = var_["PhiCor_UPhi"];
  mvaOutputCovU1_ = GetResponse(mvaReaderCovU1_, varForCovU1_)* mvaOutputU_ * var_["particleFlow_U"];
  mvaOutputCovU2_ = GetResponse(mvaReaderCovU2_, varForCovU2_)* mvaOutputU_ * var_["particleFlow_U"];


  // compute MET(Photon check)
  if(hasPhotons_) { 
    //Fix events with unphysical properties
    double sumLeptonPt = std::max(sqrt(sumLeptonPx_*sumLeptonPx_+sumLeptonPy_*sumLeptonPy_),1.);
    if(var_["track_U"]/sumLeptonPt < 0.1 || var_["noPileUp_U"]/sumLeptonPt <  0.1 ) {
      mvaOutputU_      = 1.;
      mvaOutputDPhi_   = 0.;
    }
  }
  computeMET();
}
//-------------------------------------------------------------------------------

void PFMETAlgorithmMVA::computeMET()
{
    double U      = var_["RecoilCor_U"];
    double Phi    = var_["PhiCor_UPhi"];
    if ( U < 0. ) Phi += Pi; //RF: No sign flip for U necessary in that case?
    double cosPhi = std::cos(Phi);
    double sinPhi = std::sin(Phi);
    double metPx  = U*cosPhi - sumLeptonPx_;
    double metPy  = U*sinPhi - sumLeptonPy_;
    double metPt  = sqrt(metPx*metPx + metPy*metPy);
    mvaMEt_.SetCoordinates(metPx, metPy, 0., metPt);
    // compute MET uncertainties in dirrections parallel and perpendicular to hadronic recoil
    // (neglecting uncertainties on lepton momenta)
    mvaMEtCov_(0, 0) =  mvaOutputCovU1_*cosPhi*cosPhi + mvaOutputCovU2_*sinPhi*sinPhi;
    mvaMEtCov_(0, 1) = -mvaOutputCovU1_*sinPhi*cosPhi + mvaOutputCovU2_*sinPhi*cosPhi;
    mvaMEtCov_(1, 0) =  mvaMEtCov_(0, 1);
    mvaMEtCov_(1, 1) =  mvaOutputCovU1_*sinPhi*sinPhi + mvaOutputCovU2_*cosPhi*cosPhi;
}

//-------------------------------------------------------------------------------
const float PFMETAlgorithmMVA::GetResponse(const GBRForest * Reader, std::vector<std::string> &variableNames )
{
    std::unique_ptr<float[]> mvaInputVector = createFloatVector(variableNames);
    double result = Reader->GetResponse( mvaInputVector.get() );
    return result;
}

//-------------------------------------------------------------------------------
void PFMETAlgorithmMVA::print(std::ostream& stream) const
{
  stream << "<PFMETAlgorithmMVA::print>:" << std::endl;
    for(auto entry: var_)
        stream << entry.first << " = " << entry.second << std::endl;
  stream << " covU1 = " << mvaOutputCovU1_ << ", covU2 = " << mvaOutputCovU2_ << std::endl;
  stream << " sum(leptons): Pt = " << sqrt(sumLeptonPx_*sumLeptonPx_ + sumLeptonPy_*sumLeptonPy_) << ","
  << " phi = " << atan2(sumLeptonPy_, sumLeptonPx_) << " "
  << "(Px = " << sumLeptonPx_ << ", Py = " << sumLeptonPy_ << ")" ;
  stream << " MVA output: U = " << mvaOutputU_ << ", dPhi = " << mvaOutputDPhi_ << "," << " covU1 = " << mvaOutputCovU1_ << ", covU2 = " << mvaOutputCovU2_ << std::endl;
  stream << std::endl;
}

