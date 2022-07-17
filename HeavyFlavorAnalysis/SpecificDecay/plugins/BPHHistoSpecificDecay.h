#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHHistoSpecificDecay_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHHistoSpecificDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <string>

class TH1F;
class TTree;
class TBranch;
class TVector3;

namespace reco {
  class Candidate;
  class Vertex;
}  // namespace reco

class BPHHistoSpecificDecay : public BPHAnalyzerWrapper<BPHModuleWrapper::one_analyzer> {
public:
  explicit BPHHistoSpecificDecay(const edm::ParameterSet& ps);
  ~BPHHistoSpecificDecay() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override;
  void analyze(const edm::Event& ev, const edm::EventSetup& es) override;
  void endJob() override;

  class CandidateSelect {
  public:
    virtual ~CandidateSelect() = default;
    virtual bool accept(const pat::CompositeCandidate& cand, const reco::Vertex* pv = nullptr) const = 0;
  };

private:
  std::string trigResultsLabel;
  std::string oniaCandsLabel;
  std::string sdCandsLabel;
  std::string ssCandsLabel;
  std::string buCandsLabel;
  std::string bdCandsLabel;
  std::string bsCandsLabel;
  std::string k0CandsLabel;
  std::string l0CandsLabel;
  std::string b0CandsLabel;
  std::string lbCandsLabel;
  std::string bcCandsLabel;
  std::string x3872CandsLabel;
  BPHTokenWrapper<edm::TriggerResults> trigResultsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > oniaCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > sdCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > ssCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > buCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > bdCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > bsCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > k0CandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > l0CandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > b0CandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > lbCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > bcCandsToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > x3872CandsToken;
  bool useTrig;
  bool useOnia;
  bool useSd;
  bool useSs;
  bool useBu;
  bool useBd;
  bool useBs;
  bool useK0;
  bool useL0;
  bool useB0;
  bool useLb;
  bool useBc;
  bool useX3872;

  edm::Service<TFileService> fs;
  std::map<std::string, TH1F*> histoMap;
  TTree* tree;
  unsigned int runNumber;
  unsigned int lumiSection;
  unsigned int eventNumber;
  std::string* recoName;
  float recoMass;
  float recoTime;
  float recoErrT;
  TBranch* b_runNumber;
  TBranch* b_lumiSection;
  TBranch* b_eventNumber;
  TBranch* b_recoName;
  TBranch* b_recoMass;
  TBranch* b_recoTime;
  TBranch* b_recoErrT;

  CandidateSelect* phiIBasicSelect;
  CandidateSelect* jPsiIBasicSelect;
  CandidateSelect* psi2IBasicSelect;
  CandidateSelect* upsIBasicSelect;
  CandidateSelect* phiBBasicSelect;
  CandidateSelect* jPsiBBasicSelect;
  CandidateSelect* psi2BBasicSelect;
  CandidateSelect* upsBBasicSelect;
  CandidateSelect* oniaVertexSelect;
  CandidateSelect* oniaDaughterSelect;

  CandidateSelect* npJPsiBasicSelect;
  CandidateSelect* npJPsiDaughterSelect;

  CandidateSelect* buIBasicSelect;
  CandidateSelect* buIJPsiBasicSelect;
  CandidateSelect* buIVertexSelect;
  CandidateSelect* buIJPsiDaughterSelect;
  CandidateSelect* buDBasicSelect;
  CandidateSelect* buDJPsiBasicSelect;
  CandidateSelect* buDVertexSelect;
  CandidateSelect* buDJPsiDaughterSelect;

  CandidateSelect* bdIBasicSelect;
  CandidateSelect* bdIJPsiBasicSelect;
  CandidateSelect* bdIKx0BasicSelect;
  CandidateSelect* bdIVertexSelect;
  CandidateSelect* bdIJPsiDaughterSelect;
  CandidateSelect* bdDBasicSelect;
  CandidateSelect* bdDJPsiBasicSelect;
  CandidateSelect* bdDKx0BasicSelect;
  CandidateSelect* bdDVertexSelect;
  CandidateSelect* bdDJPsiDaughterSelect;

  CandidateSelect* bsIBasicSelect;
  CandidateSelect* bsIJPsiBasicSelect;
  CandidateSelect* bsIPhiBasicSelect;
  CandidateSelect* bsIVertexSelect;
  CandidateSelect* bsIJPsiDaughterSelect;
  CandidateSelect* bsDBasicSelect;
  CandidateSelect* bsDJPsiBasicSelect;
  CandidateSelect* bsDPhiBasicSelect;
  CandidateSelect* bsDVertexSelect;
  CandidateSelect* bsDJPsiDaughterSelect;

  CandidateSelect* b0IBasicSelect;
  CandidateSelect* b0IJPsiBasicSelect;
  CandidateSelect* b0IK0sBasicSelect;
  CandidateSelect* b0IVertexSelect;
  CandidateSelect* b0IJPsiDaughterSelect;
  CandidateSelect* b0DBasicSelect;
  CandidateSelect* b0DJPsiBasicSelect;
  CandidateSelect* b0DK0sBasicSelect;
  CandidateSelect* b0DVertexSelect;
  CandidateSelect* b0DJPsiDaughterSelect;

  CandidateSelect* lbIBasicSelect;
  CandidateSelect* lbIJPsiBasicSelect;
  CandidateSelect* lbILambda0BasicSelect;
  CandidateSelect* lbIVertexSelect;
  CandidateSelect* lbIJPsiDaughterSelect;
  CandidateSelect* lbDBasicSelect;
  CandidateSelect* lbDJPsiBasicSelect;
  CandidateSelect* lbDLambda0BasicSelect;
  CandidateSelect* lbDVertexSelect;
  CandidateSelect* lbDJPsiDaughterSelect;

  CandidateSelect* bcIBasicSelect;
  CandidateSelect* bcIJPsiBasicSelect;
  CandidateSelect* bcIJPsiVertexSelect;
  CandidateSelect* bcIVertexSelect;
  CandidateSelect* bcIJPsiDaughterSelect;
  CandidateSelect* bcDBasicSelect;
  CandidateSelect* bcDJPsiBasicSelect;
  CandidateSelect* bcDJPsiVertexSelect;
  CandidateSelect* bcDVertexSelect;
  CandidateSelect* bcDJPsiDaughterSelect;

  CandidateSelect* x3872IBasicSelect;
  CandidateSelect* x3872IJPsiBasicSelect;
  CandidateSelect* x3872IJPsiVertexSelect;
  CandidateSelect* x3872IVertexSelect;
  CandidateSelect* x3872IJPsiDaughterSelect;
  CandidateSelect* x3872DBasicSelect;
  CandidateSelect* x3872DJPsiBasicSelect;
  CandidateSelect* x3872DJPsiVertexSelect;
  CandidateSelect* x3872DVertexSelect;
  CandidateSelect* x3872DJPsiDaughterSelect;

  double buIKPtMin;
  double buDKPtMin;
  double bcIPiPtMin;
  double bcDPiPtMin;
  double x3872IPiPtMin;
  double x3872DPiPtMin;
  double bcJPsiDcaMax;
  double x3872JPsiDcaMax;

  void fillHisto(const std::string& name, const pat::CompositeCandidate& cand, char svType);
  void fillHisto(const std::string& name, float x);
  void createHisto(const std::string& name, int nbin, float hmin, float hmax);
};

#endif
