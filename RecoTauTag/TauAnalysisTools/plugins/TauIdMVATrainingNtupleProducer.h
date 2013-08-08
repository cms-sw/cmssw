#ifndef RecoTauTag_TauAnalysisTools_TauIdMVATrainingNtupleProducer_h  
#define RecoTauTag_TauAnalysisTools_TauIdMVATrainingNtupleProducer_h 

/** \class TauIdMVATrainingNtupleProducer
 *
 * Produce an Ntuple containing input variables
 * for training tau isolation MVA
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: TauIdMVATrainingNtupleProducer.h,v 1.3 2012/03/08 10:31:49 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"

#include <TTree.h>
#include <TMatrixD.h>
#include <TString.h>

#include <map>
#include <string>
#include <vector>
#include <ostream>

class TauIdMVATrainingNtupleProducer : public edm::EDProducer 
{
 public:
  
  TauIdMVATrainingNtupleProducer(const edm::ParameterSet&);
  ~TauIdMVATrainingNtupleProducer();

  void produce(edm::Event&, const edm::EventSetup&);
  void beginJob();

 private:

  void setRecTauValues(const reco::PFTauRef&, const edm::Event&);
  void setGenTauMatchValues(const reco::Candidate::LorentzVector&, const reco::GenParticle*, const reco::Candidate::LorentzVector&, int);
  void setGenParticleMatchValues(const std::string&, const reco::Candidate::LorentzVector&, const reco::GenParticle*);
  void setNumPileUpValue(const edm::Event&);

  void addBranchF(const std::string&);
  void addBranchI(const std::string&);

  void addBranch_EnPxPyPz(const std::string&);
  void addBranch_XYZ(const std::string&);
  void addBranch_Cov(const std::string&);
  void addBranch_chargedHadron(const std::string&);
  void addBranch_piZero(const std::string&);
    
  void printBranches(std::ostream&);

  void setValueF(const std::string&, double);
  void setValueI(const std::string&, int);

  void setValue_EnPxPyPz(const std::string&, const reco::Candidate::LorentzVector&);
  template <typename T>
  void setValue_XYZ(const std::string&, const T&);
  void setValue_Cov(const std::string&, const reco::PFTauTransverseImpactParameter::CovMatrix&);
  void setValue_chargedHadron(const std::string&, const reco::PFRecoTauChargedHadron*);
  void setValue_piZero(const std::string&, const reco::RecoTauPiZero*);

  std::string moduleLabel_;

  edm::InputTag srcRecTaus_;
  edm::InputTag srcRecTauTransverseImpactParameters_;

  edm::InputTag srcGenTauJets_;
  edm::InputTag srcGenParticles_;
  double minGenVisPt_;
  double dRmatch_;

  unsigned maxChargedHadrons_;
  unsigned maxPiZeros_;

  struct tauIdDiscrEntryType
  {
    tauIdDiscrEntryType(const std::string& name, const edm::InputTag& src)
      : src_(src)
    {
      branchName_ = name;
    }
    ~tauIdDiscrEntryType() {}
    edm::InputTag src_;
    std::string branchName_;
  };
  std::vector<tauIdDiscrEntryType> tauIdDiscrEntries_;

  struct tauIsolationEntryType
  {
    tauIsolationEntryType(const std::string& name, const edm::ParameterSet& cfg)
      : srcChargedIsoPtSum_(cfg.getParameter<edm::InputTag>("chargedIsoPtSum")),
	srcNeutralIsoPtSum_(cfg.getParameter<edm::InputTag>("neutralIsoPtSum")),
	srcPUcorrPtSum_(cfg.getParameter<edm::InputTag>("puCorrPtSum"))
    {
      branchNameChargedIsoPtSum_ = Form("%sChargedIsoPtSum", name.data());
      branchNameNeutralIsoPtSum_ = Form("%sNeutralIsoPtSum", name.data());
      branchNamePUcorrPtSum_     = Form("%sPUcorrPtSum", name.data());
    }
    ~tauIsolationEntryType() {}
    edm::InputTag srcChargedIsoPtSum_;
    std::string branchNameChargedIsoPtSum_;
    edm::InputTag srcNeutralIsoPtSum_;
    std::string branchNameNeutralIsoPtSum_;
    edm::InputTag srcPUcorrPtSum_;
    std::string branchNamePUcorrPtSum_;
  };
  std::vector<tauIsolationEntryType> tauIsolationEntries_;

  struct vertexCollectionEntryType
  {
    vertexCollectionEntryType(const std::string& name, const edm::InputTag& src)
      : src_(src)
    {
      assert(name.length() > 0);
      std::string name_capitalized = name;
      name_capitalized[0] = toupper(name_capitalized[0]);
      branchName_multiplicity_ = Form("num%s", name_capitalized.data());
      branchName_position_ = TString(name.data()).ReplaceAll("Vertices", "Vertex").Data();      
    }
    ~vertexCollectionEntryType() {}
    edm::InputTag src_;
    std::string branchName_multiplicity_;
    std::string branchName_position_;
  };
  std::vector<vertexCollectionEntryType> vertexCollectionEntries_;

  std::vector<int> pdgIdsGenTau_;
  std::vector<int> pdgIdsGenElectron_;
  std::vector<int> pdgIdsGenMuon_;
  std::vector<int> pdgIdsGenQuarkOrGluon_;
  
  PFJetIDSelectionFunctor* loosePFJetIdAlgo_;

  bool isMC_;
  edm::InputTag srcGenPileUpSummary_;
  std::map<edm::RunNumber_t, std::map<edm::LuminosityBlockNumber_t, float> > pileUpByLumiCalc_; // key = run, lumi-section
  std::map<edm::RunNumber_t, std::map<edm::LuminosityBlockNumber_t, int> > numWarnings_;
  int maxWarnings_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcWeights_;

  struct branchEntryType
  {
    branchEntryType()
      : valueF_(0.),
	valueI_(0)
    {}
    ~branchEntryType() {}
    Float_t valueF_;
    Int_t valueI_;
  };
  typedef std::map<std::string, branchEntryType> branchMap; // key = branch name
  branchMap branches_;

  TTree* ntuple_;

  int verbosity_;
};

#endif


