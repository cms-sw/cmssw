#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleGen : public HGCalTriggerNtupleBase
{

    public:
        HGCalTriggerNtupleGen(const edm::ParameterSet& );

        virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
        virtual void fill(const edm::Event&, const edm::EventSetup& ) override final;

    private:
        virtual void clear() override final;

        edm::EDGetToken gen_token_;
        int gen_n_;
        std::vector<int>   gen_id_;
        std::vector<int>   gen_status_;
        std::vector<float> gen_energy_;
        std::vector<float> gen_pt_;
        std::vector<float> gen_eta_;
        std::vector<float> gen_phi_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
        HGCalTriggerNtupleGen,
        "HGCalTriggerNtupleGen" );


HGCalTriggerNtupleGen::
HGCalTriggerNtupleGen(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleGen::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{

    gen_token_ = collector.consumes<reco::GenParticleCollection>(conf.getParameter<edm::InputTag>("GenParticles"));
    tree.Branch("gen_n", &gen_n_, "gen_n/I");
    tree.Branch("gen_id", &gen_id_);
    tree.Branch("gen_status", &gen_status_);
    tree.Branch("gen_energy", &gen_energy_);
    tree.Branch("gen_pt", &gen_pt_);
    tree.Branch("gen_eta", &gen_eta_);
    tree.Branch("gen_phi", &gen_phi_);

}

void
HGCalTriggerNtupleGen::
fill(const edm::Event& e, const edm::EventSetup& es)
{
    edm::Handle<reco::GenParticleCollection> gen_particles_h;
    e.getByToken(gen_token_, gen_particles_h);
    const reco::GenParticleCollection& gen_particles = *gen_particles_h;

    clear();
    gen_n_ = gen_particles.size();
    gen_id_.reserve(gen_n_);
    gen_status_.reserve(gen_n_);
    gen_energy_.reserve(gen_n_);
    gen_pt_.reserve(gen_n_);
    gen_eta_.reserve(gen_n_);
    gen_phi_.reserve(gen_n_);
    for(const auto& particle : gen_particles)
    {
        gen_id_.emplace_back(particle.pdgId());
        gen_status_.emplace_back(particle.status());
        gen_energy_.emplace_back(particle.energy());
        gen_pt_.emplace_back(particle.pt());
        gen_eta_.emplace_back(particle.eta());
        gen_phi_.emplace_back(particle.phi());
    }

}


void
HGCalTriggerNtupleGen::
clear()
{
    gen_n_ = 0;
    gen_id_.clear();
    gen_status_.clear();
    gen_energy_.clear();
    gen_pt_.clear();
    gen_eta_.clear();
    gen_phi_.clear();
}




