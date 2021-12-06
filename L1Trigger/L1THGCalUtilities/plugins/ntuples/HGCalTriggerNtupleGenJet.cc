#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"

class HGCalTriggerNtupleGenJet : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleGenJet(const edm::ParameterSet&);

  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event&, const HGCalTriggerNtupleEventSetup&) final;

private:
  void clear() final;

  edm::EDGetToken genjet_token_;

  int genjet_n_;
  std::vector<float> genjet_energy_;
  std::vector<float> genjet_pt_;
  std::vector<float> genjet_eta_;
  std::vector<float> genjet_phi_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleGenJet, "HGCalTriggerNtupleGenJet");

HGCalTriggerNtupleGenJet::HGCalTriggerNtupleGenJet(const edm::ParameterSet& conf) : HGCalTriggerNtupleBase(conf) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleGenJet::initialize(TTree& tree,
                                          const edm::ParameterSet& conf,
                                          edm::ConsumesCollector&& collector) {
  genjet_token_ = collector.consumes<reco::GenJetCollection>(conf.getParameter<edm::InputTag>("GenJets"));
  tree.Branch("genjet_n", &genjet_n_, "genjet_n/I");
  tree.Branch("genjet_energy", &genjet_energy_);
  tree.Branch("genjet_pt", &genjet_pt_);
  tree.Branch("genjet_eta", &genjet_eta_);
  tree.Branch("genjet_phi", &genjet_phi_);
}

void HGCalTriggerNtupleGenJet::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  edm::Handle<reco::GenJetCollection> genjets_h;
  e.getByToken(genjet_token_, genjets_h);
  const reco::GenJetCollection& genjets = *genjets_h;

  clear();
  genjet_n_ = genjets.size();
  genjet_energy_.reserve(genjet_n_);
  genjet_pt_.reserve(genjet_n_);
  genjet_eta_.reserve(genjet_n_);
  genjet_phi_.reserve(genjet_n_);
  for (const auto& jet : genjets) {
    genjet_energy_.emplace_back(jet.energy());
    genjet_pt_.emplace_back(jet.pt());
    genjet_eta_.emplace_back(jet.eta());
    genjet_phi_.emplace_back(jet.phi());
  }
}

void HGCalTriggerNtupleGenJet::clear() {
  genjet_n_ = 0;
  genjet_energy_.clear();
  genjet_pt_.clear();
  genjet_eta_.clear();
  genjet_phi_.clear();
}
