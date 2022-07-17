#include <vector>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "DataFormats/Math/interface/LorentzVector.h"

typedef math::XYZTLorentzVector LorentzVector;

class HGCalTriggerNtupleGenTau : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleGenTau(const edm::ParameterSet&);

  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event&, const HGCalTriggerNtupleEventSetup&) final;

private:
  void clear() final;

  bool isGoodTau(const reco::GenParticle& candidate) const;
  bool isStableLepton(const reco::GenParticle& daughter) const;
  bool isElectron(const reco::GenParticle& daughter) const;
  bool isMuon(const reco::GenParticle& daughter) const;
  bool isChargedHadron(const reco::GenParticle& daughter) const;
  bool isChargedHadronFromResonance(const reco::GenParticle& daughter) const;
  bool isNeutralPion(const reco::GenParticle& daughter) const;
  bool isNeutralPionFromResonance(const reco::GenParticle& daughter) const;
  bool isIntermediateResonance(const reco::GenParticle& daughter) const;
  bool isGamma(const reco::GenParticle& daughter) const;
  bool isStableNeutralHadron(const reco::GenParticle& daughter) const;

  edm::EDGetToken gen_token_;
  bool isPythia8generator_;

  std::vector<float> gentau_pt_;
  std::vector<float> gentau_eta_;
  std::vector<float> gentau_phi_;
  std::vector<float> gentau_energy_;
  std::vector<float> gentau_mass_;

  std::vector<float> gentau_vis_pt_;
  std::vector<float> gentau_vis_eta_;
  std::vector<float> gentau_vis_phi_;
  std::vector<float> gentau_vis_energy_;
  std::vector<float> gentau_vis_mass_;
  std::vector<int> gentau_decayMode_;
  std::vector<int> gentau_totNproducts_;
  std::vector<int> gentau_totNgamma_;
  std::vector<int> gentau_totNpiZero_;
  std::vector<int> gentau_totNcharged_;

  std::vector<std::vector<float> > gentau_products_pt_;
  std::vector<std::vector<float> > gentau_products_eta_;
  std::vector<std::vector<float> > gentau_products_phi_;
  std::vector<std::vector<float> > gentau_products_energy_;
  std::vector<std::vector<float> > gentau_products_mass_;
  std::vector<std::vector<int> > gentau_products_id_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleGenTau, "HGCalTriggerNtupleGenTau");

HGCalTriggerNtupleGenTau::HGCalTriggerNtupleGenTau(const edm::ParameterSet& conf) : HGCalTriggerNtupleBase(conf) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleGenTau::initialize(TTree& tree,
                                          const edm::ParameterSet& conf,
                                          edm::ConsumesCollector&& collector) {
  gen_token_ = collector.consumes<reco::GenParticleCollection>(conf.getParameter<edm::InputTag>("GenParticles"));
  isPythia8generator_ = conf.getParameter<bool>("isPythia8");

  tree.Branch("gentau_pt", &gentau_pt_);
  tree.Branch("gentau_eta", &gentau_eta_);
  tree.Branch("gentau_phi", &gentau_phi_);
  tree.Branch("gentau_energy", &gentau_energy_);
  tree.Branch("gentau_mass", &gentau_mass_);
  tree.Branch("gentau_vis_pt", &gentau_vis_pt_);
  tree.Branch("gentau_vis_eta", &gentau_vis_eta_);
  tree.Branch("gentau_vis_phi", &gentau_vis_phi_);
  tree.Branch("gentau_vis_energy", &gentau_vis_energy_);
  tree.Branch("gentau_vis_mass", &gentau_vis_mass_);
  tree.Branch("gentau_products_pt", &gentau_products_pt_);
  tree.Branch("gentau_products_eta", &gentau_products_eta_);
  tree.Branch("gentau_products_phi", &gentau_products_phi_);
  tree.Branch("gentau_products_energy", &gentau_products_energy_);
  tree.Branch("gentau_products_mass", &gentau_products_mass_);
  tree.Branch("gentau_products_id", &gentau_products_id_);
  tree.Branch("gentau_decayMode", &gentau_decayMode_);
  tree.Branch("gentau_totNproducts", &gentau_totNproducts_);
  tree.Branch("gentau_totNgamma", &gentau_totNgamma_);
  tree.Branch("gentau_totNpiZero", &gentau_totNpiZero_);
  tree.Branch("gentau_totNcharged", &gentau_totNcharged_);
}

bool HGCalTriggerNtupleGenTau::isGoodTau(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 15 && candidate.status() == 2);
}

bool HGCalTriggerNtupleGenTau::isChargedHadron(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 211 || std::abs(candidate.pdgId()) == 321) && candidate.status() == 1 &&
          candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool HGCalTriggerNtupleGenTau::isChargedHadronFromResonance(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 211 || std::abs(candidate.pdgId()) == 321) && candidate.status() == 1 &&
          candidate.isLastCopy());
}

bool HGCalTriggerNtupleGenTau::isStableLepton(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 11 || std::abs(candidate.pdgId()) == 13) && candidate.status() == 1 &&
          candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool HGCalTriggerNtupleGenTau::isElectron(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 11 && candidate.isDirectPromptTauDecayProductFinalState() &&
          candidate.isLastCopy());
}

bool HGCalTriggerNtupleGenTau::isMuon(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 13 && candidate.isDirectPromptTauDecayProductFinalState() &&
          candidate.isLastCopy());
}

bool HGCalTriggerNtupleGenTau::isNeutralPion(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 111 && candidate.status() == 2 &&
          candidate.statusFlags().isTauDecayProduct() && !candidate.isDirectPromptTauDecayProductFinalState());
}

bool HGCalTriggerNtupleGenTau::isNeutralPionFromResonance(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 111 && candidate.status() == 2 && candidate.statusFlags().isTauDecayProduct());
}

bool HGCalTriggerNtupleGenTau::isGamma(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 22 && candidate.status() == 1 && candidate.statusFlags().isTauDecayProduct() &&
          !candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool HGCalTriggerNtupleGenTau::isIntermediateResonance(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 213 || std::abs(candidate.pdgId()) == 20213 ||
           std::abs(candidate.pdgId()) == 24) &&
          candidate.status() == 2);
}

bool HGCalTriggerNtupleGenTau::isStableNeutralHadron(const reco::GenParticle& candidate) const {
  return (!(std::abs(candidate.pdgId()) > 10 && std::abs(candidate.pdgId()) < 17) && !isChargedHadron(candidate) &&
          candidate.status() == 1);
}

void HGCalTriggerNtupleGenTau::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  edm::Handle<reco::GenParticleCollection> gen_particles_h;
  e.getByToken(gen_token_, gen_particles_h);
  const reco::GenParticleCollection& gen_particles = *gen_particles_h;

  clear();

  for (const auto& particle : gen_particles) {
    /* select good taus */
    if (isGoodTau(particle)) {
      LorentzVector tau_p4vis(0., 0., 0., 0.);
      gentau_pt_.emplace_back(particle.pt());
      gentau_eta_.emplace_back(particle.eta());
      gentau_phi_.emplace_back(particle.phi());
      gentau_energy_.emplace_back(particle.energy());
      gentau_mass_.emplace_back(particle.mass());

      int n_pi = 0;
      int n_piZero = 0;
      int n_gamma = 0;
      int n_ele = 0;
      int n_mu = 0;

      std::vector<float> tau_products_pt;
      std::vector<float> tau_products_eta;
      std::vector<float> tau_products_phi;
      std::vector<float> tau_products_energy;
      std::vector<float> tau_products_mass;
      std::vector<int> tau_products_id;

      /* loop over tau daughters */
      const reco::GenParticleRefVector& daughters = particle.daughterRefVector();

      for (const auto& daughter : daughters) {
        reco::GenParticleRefVector finalProds;

        if (isStableLepton(*daughter)) {
          if (isElectron(*daughter)) {
            n_ele++;
          } else if (isMuon(*daughter)) {
            n_mu++;
          }
          tau_p4vis += (daughter->p4());
          finalProds.push_back(daughter);
        }

        else if (isChargedHadron(*daughter)) {
          n_pi++;
          tau_p4vis += (daughter->p4());
          finalProds.push_back(daughter);
        }

        else if (isNeutralPion(*daughter)) {
          n_piZero++;
          const reco::GenParticleRefVector& granddaughters = daughter->daughterRefVector();
          for (const auto& granddaughter : granddaughters) {
            if (isGamma(*granddaughter)) {
              n_gamma++;
              tau_p4vis += (granddaughter->p4());
              finalProds.push_back(granddaughter);
            }
          }
        }

        else if (isStableNeutralHadron(*daughter)) {
          tau_p4vis += (daughter->p4());
          finalProds.push_back(daughter);
        }

        else {
          const reco::GenParticleRefVector& granddaughters = daughter->daughterRefVector();

          for (const auto& granddaughter : granddaughters) {
            if (isStableNeutralHadron(*granddaughter)) {
              tau_p4vis += (granddaughter->p4());
              finalProds.push_back(granddaughter);
            }
          }
        }

        /* Here the selection of the decay product according to the Pythia6 decayTree */
        if (!isPythia8generator_) {
          if (isIntermediateResonance(*daughter)) {
            const reco::GenParticleRefVector& grandaughters = daughter->daughterRefVector();
            for (const auto& grandaughter : grandaughters) {
              if (isChargedHadron(*grandaughter) || isChargedHadronFromResonance(*grandaughter)) {
                n_pi++;
                tau_p4vis += (grandaughter->p4());
                finalProds.push_back(daughter);
              } else if (isNeutralPion(*grandaughter) || isNeutralPionFromResonance(*grandaughter)) {
                n_piZero++;
                const reco::GenParticleRefVector& descendants = grandaughter->daughterRefVector();
                for (const auto& descendant : descendants) {
                  if (isGamma(*descendant)) {
                    n_gamma++;
                    tau_p4vis += (descendant->p4());
                    finalProds.push_back(daughter);
                  }
                }
              }
            }
          }
        }

        /* Fill daughter informations */
        for (const auto& prod : finalProds) {
          tau_products_pt.emplace_back(prod->pt());
          tau_products_eta.emplace_back(prod->eta());
          tau_products_phi.emplace_back(prod->phi());
          tau_products_energy.emplace_back(prod->energy());
          tau_products_mass.emplace_back(prod->mass());
          tau_products_id.emplace_back(prod->pdgId());
        }
      }

      /* assign the tau-variables */
      gentau_vis_pt_.emplace_back(tau_p4vis.Pt());
      gentau_vis_eta_.emplace_back(tau_p4vis.Eta());
      gentau_vis_phi_.emplace_back(tau_p4vis.Phi());
      gentau_vis_energy_.emplace_back(tau_p4vis.E());
      gentau_vis_mass_.emplace_back(tau_p4vis.M());
      gentau_totNproducts_.emplace_back(n_pi + n_gamma);
      gentau_totNgamma_.emplace_back(n_gamma);
      gentau_totNpiZero_.emplace_back(n_piZero);
      gentau_totNcharged_.emplace_back(n_pi);

      gentau_products_pt_.emplace_back(tau_products_pt);
      gentau_products_eta_.emplace_back(tau_products_eta);
      gentau_products_phi_.emplace_back(tau_products_phi);
      gentau_products_energy_.emplace_back(tau_products_energy);
      gentau_products_mass_.emplace_back(tau_products_mass);
      gentau_products_id_.emplace_back(tau_products_id);

      /* leptonic tau decays */
      if (n_pi == 0 && n_piZero == 0 && n_ele == 1) {
        gentau_decayMode_.emplace_back(11);
      } else if (n_pi == 0 && n_piZero == 0 && n_mu == 1) {
        gentau_decayMode_.emplace_back(13);
      }
      /* 1-prong */
      else if (n_pi == 1 && n_piZero == 0) {
        gentau_decayMode_.emplace_back(0);
      }
      /* 1-prong + pi0s */
      else if (n_pi == 1 && n_piZero >= 1) {
        gentau_decayMode_.emplace_back(1);
      }
      /* 3-prongs */
      else if (n_pi == 3 && n_piZero == 0) {
        gentau_decayMode_.emplace_back(4);
      }
      /* 3-prongs + pi0s */
      else if (n_pi == 3 && n_piZero >= 1) {
        gentau_decayMode_.emplace_back(5);
      }
      /* other decays */
      else {
        gentau_decayMode_.emplace_back(-1);
      }
    }
  }
}

void HGCalTriggerNtupleGenTau::clear() {
  gentau_pt_.clear();
  gentau_eta_.clear();
  gentau_phi_.clear();
  gentau_energy_.clear();
  gentau_mass_.clear();
  gentau_decayMode_.clear();
  gentau_vis_pt_.clear();
  gentau_vis_eta_.clear();
  gentau_vis_phi_.clear();
  gentau_vis_energy_.clear();
  gentau_vis_mass_.clear();
  gentau_totNproducts_.clear();
  gentau_totNgamma_.clear();
  gentau_totNcharged_.clear();
  gentau_products_pt_.clear();
  gentau_products_eta_.clear();
  gentau_products_phi_.clear();
  gentau_products_energy_.clear();
  gentau_products_mass_.clear();
  gentau_products_id_.clear();
}
