#include <vector>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
#include "Math/LorentzVector.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > LorentzVector;


class HGCalTriggerNtupleGenTau : public HGCalTriggerNtupleBase
{

    public:
        HGCalTriggerNtupleGenTau(const edm::ParameterSet& );

        virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
        virtual void fill(const edm::Event&, const edm::EventSetup& ) override final;

    private:
        virtual void clear() override final;

        edm::EDGetToken gen_token_;
        bool isPythia8generator_;

        std::vector<float> gen_tau_pt_;
        std::vector<float> gen_tau_eta_;
        std::vector<float> gen_tau_phi_;
        std::vector<float> gen_tau_energy_;
        std::vector<float> gen_tau_mass_;

        std::vector<float> gen_tauVis_pt_;
        std::vector<float> gen_tauVis_eta_;
        std::vector<float> gen_tauVis_phi_;
        std::vector<float> gen_tauVis_energy_;
        std::vector<float> gen_tauVis_mass_;    
        std::vector<int> gen_tau_decayMode_;
        std::vector<int> gen_tau_totNproducts_;
        std::vector<int> gen_tau_totNgamma_;
        std::vector<int> gen_tau_totNcharged_;

        std::vector<std::vector<float> > gen_product_pt_;
        std::vector<std::vector<float> > gen_product_eta_;
        std::vector<std::vector<float> > gen_product_phi_;
        std::vector<std::vector<float> > gen_product_energy_;
        std::vector<std::vector<float> > gen_product_mass_;
        std::vector<std::vector< int > > gen_product_id_;
        
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
        HGCalTriggerNtupleGenTau,
        "HGCalTriggerNtupleGenTau" );


HGCalTriggerNtupleGenTau::
HGCalTriggerNtupleGenTau(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleGenTau::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{

    gen_token_ = collector.consumes<reco::GenParticleCollection>(conf.getParameter<edm::InputTag>("GenParticles"));
    isPythia8generator_ = conf.getParameter<bool>("isPythia8");

    tree.Branch("gen_tau_pt", &gen_tau_pt_);
    tree.Branch("gen_tau_eta", &gen_tau_eta_);
    tree.Branch("gen_tau_phi", &gen_tau_phi_);
    tree.Branch("gen_tau_energy", &gen_tau_energy_);
    tree.Branch("gen_tau_mass", &gen_tau_mass_);
    tree.Branch("gen_tauVis_pt", &gen_tauVis_pt_);
    tree.Branch("gen_tauVis_eta", &gen_tauVis_eta_);
    tree.Branch("gen_tauVis_phi", &gen_tauVis_phi_);
    tree.Branch("gen_tauVis_energy", &gen_tauVis_energy_);
    tree.Branch("gen_tauVis_mass", &gen_tauVis_mass_);
    tree.Branch("gen_product_pt", &gen_product_pt_);
    tree.Branch("gen_product_eta", &gen_product_eta_);
    tree.Branch("gen_product_phi", &gen_product_phi_);
    tree.Branch("gen_product_energy", &gen_product_energy_);
    tree.Branch("gen_product_mass", &gen_product_mass_);
    tree.Branch("gen_product_id", &gen_product_id_);
    tree.Branch("gen_tau_decayMode", &gen_tau_decayMode_);
    tree.Branch("gen_tau_totNproducts", &gen_tau_totNproducts_);
    tree.Branch("gen_tau_totNgamma", &gen_tau_totNgamma_);
    tree.Branch("gen_tau_totNcharged", &gen_tau_totNcharged_);

}

void
HGCalTriggerNtupleGenTau::
fill(const edm::Event& e, const edm::EventSetup& es)
{
    edm::Handle<reco::GenParticleCollection> gen_particles_h;
    e.getByToken(gen_token_, gen_particles_h);
    const reco::GenParticleCollection& gen_particles = *gen_particles_h;

    clear();
    
    for(const auto& particle : gen_particles)
    {
        
        /* select good taus */
        if(fabs(particle.pdgId())==15 && particle.status()==2){

            size_t n = particle.numberOfDaughters();
            LorentzVector tau_p4 = particle.p4();
            LorentzVector tau_p4vis(0.,0.,0.,0.);
            gen_tau_pt_.emplace_back(tau_p4.Pt());
            gen_tau_eta_.emplace_back(tau_p4.Eta());
            gen_tau_phi_.emplace_back(tau_p4.Phi());
            gen_tau_energy_.emplace_back(tau_p4.E());
            gen_tau_mass_.emplace_back(tau_p4.M());

            int n_pi=0;
            int n_piZero=0;
            int n_gamma=0;
            int n_ele=0;
            int n_mu=0;

            std::vector<float> tau_products_pt;
            std::vector<float> tau_products_eta;
            std::vector<float> tau_products_phi;
            std::vector<float> tau_products_energy;
            std::vector<float> tau_products_mass;
            std::vector< int > tau_products_id;

            /* loop over tau daughters */
            for(size_t j = 0; j < n; ++ j) {
                const reco::Candidate * daughter = particle.daughter( j );                
                 
                if( ( fabs(daughter->pdgId()) == 11 || fabs(daughter->pdgId()) == 13 ) && daughter->status()==1){
                    if( fabs(daughter->pdgId()) == 11){
                        n_ele++;
                    }
                    else{
                        n_mu++;
                    }
                    LorentzVector finalProd_p4 = daughter->p4();                      
                    tau_p4vis+=finalProd_p4;
                    tau_products_pt.emplace_back(finalProd_p4.Pt());
                    tau_products_eta.emplace_back(finalProd_p4.Eta());
                    tau_products_phi.emplace_back(finalProd_p4.Phi());
                    tau_products_energy.emplace_back(finalProd_p4.E());
                    tau_products_mass.emplace_back(finalProd_p4.M());                                    
                    tau_products_id.emplace_back(daughter->pdgId());
                }        
                
                if(isPythia8generator_){
                    if( fabs(daughter->pdgId()) == 211 && daughter->status()==1){
                        n_pi++;
                        LorentzVector finalProd_p4 = daughter->p4();                      
                        tau_p4vis+=finalProd_p4;
                        tau_products_pt.emplace_back(finalProd_p4.Pt());
                        tau_products_eta.emplace_back(finalProd_p4.Eta());
                        tau_products_phi.emplace_back(finalProd_p4.Phi());
                        tau_products_energy.emplace_back(finalProd_p4.E());
                        tau_products_mass.emplace_back(finalProd_p4.M());                                    
                        tau_products_id.emplace_back(daughter->pdgId());
                    }                
                    if( fabs(daughter->pdgId()) == 111 && daughter->status()==2 ){
                        n_piZero++;
                        size_t nGamma = daughter->numberOfDaughters();
                        for(size_t ng=0; ng<nGamma; ++ng){
                            const reco::Candidate * gamma = daughter->daughter( ng );                
                            if( fabs(gamma->pdgId())==22 && gamma->status()==1 ){
                                n_gamma++;
                                LorentzVector finalProd_p4 = gamma->p4();                      
                                tau_p4vis+=finalProd_p4;         
                                tau_products_pt.emplace_back(finalProd_p4.Pt());
                                tau_products_eta.emplace_back(finalProd_p4.Eta());
                                tau_products_phi.emplace_back(finalProd_p4.Phi());
                                tau_products_energy.emplace_back(finalProd_p4.E());
                                tau_products_mass.emplace_back(finalProd_p4.M());
                                tau_products_id.emplace_back(gamma->pdgId());    
                            }
                        }              
                    }
                }
                else if(!isPythia8generator_){            
                    if( fabs(daughter->pdgId()) == 211 || fabs(daughter->pdgId()) == 213 || fabs(daughter->pdgId()) == 20213 || fabs(daughter->pdgId()) == 24 ){
                        size_t nn = daughter->numberOfDaughters();
                        if(nn==0 && daughter->status()==1){
                            if( fabs(daughter->pdgId()) == 211 && daughter->status()==1){
                                n_pi++;
                                LorentzVector finalProd_p4 = daughter->p4();                      
                                tau_p4vis+=finalProd_p4;
                                tau_products_pt.emplace_back(finalProd_p4.Pt());
                                tau_products_eta.emplace_back(finalProd_p4.Eta());
                                tau_products_phi.emplace_back(finalProd_p4.Phi());
                                tau_products_energy.emplace_back(finalProd_p4.E());
                                tau_products_mass.emplace_back(finalProd_p4.M());                                    
                                tau_products_id.emplace_back(daughter->pdgId());
                            }
                            if( fabs(daughter->pdgId()) == 111 && daughter->status()==2 ){
                                n_piZero++;
                                size_t nGamma = daughter->numberOfDaughters();
                                for(size_t ng=0; ng<nGamma; ++ng){
                                    const reco::Candidate * gamma = daughter->daughter( ng );                
                                    if( fabs(gamma->pdgId())==22 && gamma->status()==1 ){
                                        n_gamma++;
                                        LorentzVector finalProd_p4 = gamma->p4();                      
                                        tau_p4vis+=finalProd_p4;         
                                        tau_products_pt.emplace_back(finalProd_p4.Pt());
                                        tau_products_eta.emplace_back(finalProd_p4.Eta());
                                        tau_products_phi.emplace_back(finalProd_p4.Phi());
                                        tau_products_energy.emplace_back(finalProd_p4.E());
                                        tau_products_mass.emplace_back(finalProd_p4.M());
                                        tau_products_id.emplace_back(gamma->pdgId());    
                                    }
                                }              
                            }
                        }
                        else{
                            for(size_t k = 0; k < nn; ++k) {
                                const reco::Candidate * grandson = daughter->daughter( k );
                                if( fabs(grandson->pdgId()) == 211 && grandson->status()==1 ){
                                    n_pi++;
                                    LorentzVector finalProd_p4 = grandson->p4();                      
                                    tau_p4vis+=finalProd_p4;
                                    tau_products_pt.emplace_back(finalProd_p4.Pt());
                                    tau_products_eta.emplace_back(finalProd_p4.Eta());
                                    tau_products_phi.emplace_back(finalProd_p4.Phi());
                                    tau_products_energy.emplace_back(finalProd_p4.E());
                                    tau_products_mass.emplace_back(finalProd_p4.M());
                                    tau_products_id.emplace_back(grandson->pdgId());
                                }
                                if( fabs(grandson->pdgId()) == 111 && grandson->status()==2 ){
                                    n_piZero++;
                                    size_t nGamma = grandson->numberOfDaughters();
                                    for(size_t ng=0; ng<nGamma; ++ng){
                                        const reco::Candidate * gamma = grandson->daughter( ng );
                                        if( fabs(gamma->pdgId())==22 && gamma->status()==1 ){
                                            n_gamma++;
                                            LorentzVector finalProd_p4 = gamma->p4();                      
                                            tau_p4vis+=finalProd_p4; 
                                            tau_products_pt.emplace_back(finalProd_p4.Pt());
                                            tau_products_eta.emplace_back(finalProd_p4.Eta());
                                            tau_products_phi.emplace_back(finalProd_p4.Phi());
                                            tau_products_energy.emplace_back(finalProd_p4.E());
                                            tau_products_mass.emplace_back(finalProd_p4.M());
                                            tau_products_id.emplace_back(gamma->pdgId());
                                        }
                                    }
                                }
                            }                        
                        }                    
                    }
                }  
            }/* end loop over daughters */
           
            /* assign the tau-variables */
            gen_tauVis_pt_.emplace_back(tau_p4vis.Pt());
            gen_tauVis_eta_.emplace_back(tau_p4vis.Eta());
            gen_tauVis_phi_.emplace_back(tau_p4vis.Phi());
            gen_tauVis_energy_.emplace_back(tau_p4vis.E());
            gen_tauVis_mass_.emplace_back(tau_p4vis.M());
            gen_tau_totNproducts_.emplace_back(n_pi + n_gamma);
            gen_tau_totNgamma_.emplace_back(n_gamma);
            gen_tau_totNcharged_.emplace_back(n_pi);
   
            gen_product_pt_.emplace_back(tau_products_pt);
            gen_product_eta_.emplace_back(tau_products_eta);
            gen_product_phi_.emplace_back(tau_products_phi);
            gen_product_energy_.emplace_back(tau_products_energy);
            gen_product_mass_.emplace_back(tau_products_mass);
            gen_product_id_.emplace_back(tau_products_id);

            if( n_pi == 0 && n_piZero == 0 && n_ele==1 ){ gen_tau_decayMode_.emplace_back(11); }
            else if( n_pi == 0 && n_piZero == 0 && n_mu==1 ){ gen_tau_decayMode_.emplace_back(13); }
            else if( n_pi == 1 && n_piZero == 0 ){ gen_tau_decayMode_.emplace_back(0); }
            else if( n_pi == 1 && n_piZero >= 1 ){ gen_tau_decayMode_.emplace_back(1); }
            else if( n_pi == 3 && n_piZero == 0 ){ gen_tau_decayMode_.emplace_back(4); }
            else if( n_pi == 3 && n_piZero >= 1 ){ gen_tau_decayMode_.emplace_back(5); }
            else{ gen_tau_decayMode_.emplace_back(-1); } 

        }
    }

}


void
HGCalTriggerNtupleGenTau::
clear()
{
    gen_tau_pt_.clear();
    gen_tau_eta_.clear();
    gen_tau_phi_.clear();
    gen_tau_energy_.clear();
    gen_tau_mass_.clear();
    gen_tau_decayMode_.clear();
    gen_tauVis_pt_.clear();
    gen_tauVis_eta_.clear();
    gen_tauVis_phi_.clear();
    gen_tauVis_energy_.clear();
    gen_tauVis_mass_.clear();
    gen_tau_totNproducts_.clear();
    gen_tau_totNgamma_.clear();
    gen_tau_totNcharged_.clear();
    gen_product_pt_.clear();
    gen_product_eta_.clear();
    gen_product_phi_.clear();
    gen_product_energy_.clear();
    gen_product_mass_.clear();
    gen_product_id_.clear();
}




