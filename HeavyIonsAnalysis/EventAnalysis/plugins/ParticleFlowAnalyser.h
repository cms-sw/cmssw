#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <vector>

#include "TTree.h"

class ParticleFlowAnalyser : public edm::one::EDAnalyzer<> {
public:
  explicit ParticleFlowAnalyser(const edm::ParameterSet&);
  ~ParticleFlowAnalyser() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void clear();

private:
  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;

  edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidateToken_;

  float ptMin_;
  float absEtaMax_;

  int nPF_;
  std::vector<int> pfId_;
  std::vector<float> pfPt_;
  std::vector<float> pfEta_;
  std::vector<float> pfPhi_;
  std::vector<float> pfE_;
  std::vector<float> pfM_;

  TTree* tree_;

  /* required since conversion function is not static */
  reco::PFCandidate converter_;
};
