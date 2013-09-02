#ifndef RecoTauTag_TauAnalysisTools_AntiMuonDiscrMVATrainingNtupleProducer_h  
#define RecoTauTag_TauAnalysisTools_AntiMuonDiscrMVATrainingNtupleProducer_h 

/** \class AntiMuonDiscrMVATrainingNtupleProducer
 *
 * Produce an Ntuple containing input variables
 * for training tau isolation MVA
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: AntiMuonDiscrMVATrainingNtupleProducer.h,v 1.3 2012/03/08 10:31:49 veelken Exp $
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

class AntiMuonDiscrMVATrainingNtupleProducer : public edm::EDProducer 
{
 public:
  
  AntiMuonDiscrMVATrainingNtupleProducer(const edm::ParameterSet&);
  ~AntiMuonDiscrMVATrainingNtupleProducer();

  void produce(edm::Event&, const edm::EventSetup&);
  void beginJob();

 private:

  void setRecTauValues(const reco::PFTauRef&, const edm::Event&);
  void setGenTauMatchValues(const reco::Candidate::LorentzVector&, const reco::GenParticle*, const reco::Candidate::LorentzVector&, int);
  void setGenParticleMatchValues(const std::string&, const reco::Candidate::LorentzVector&, const reco::GenParticle*);

  void addBranchF(const std::string&);
  void addBranchI(const std::string&);

  void addBranch_EnPxPyPz(const std::string&);
  void addBranch_XYZ(const std::string&);
    
  void printBranches(std::ostream&);

  void setValueF(const std::string&, double);
  void setValueI(const std::string&, int);

  void setValue_EnPxPyPz(const std::string&, const reco::Candidate::LorentzVector&);
  template <typename T>
  void setValue_XYZ(const std::string&, const T&);

  std::string moduleLabel_;

  edm::InputTag srcRecTaus_;

  edm::InputTag srcMuons_;
  edm::Handle<reco::MuonCollection> muons_;
  double dRmuonMatch_;

  edm::InputTag srcGenTauJets_;
  edm::InputTag srcGenParticles_;
  double minGenVisPt_;
  double dRgenParticleMatch_;

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
  std::vector<int> pdgIdsGenMuon_;

  PFJetIDSelectionFunctor* loosePFJetIdAlgo_;

  bool isMC_;

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


