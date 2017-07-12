#include <vector>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeRefCandidateT.h"
#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
#include "Math/LorentzVector.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > LorentzVector;


class HGCalTriggerNtupleGenTau : public HGCalTriggerNtupleBase
{

    public:
        HGCalTriggerNtupleGenTau(const edm::ParameterSet& );

        virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
        virtual void fill(const edm::Event&, const edm::EventSetup& ) override final;
    
        bool isStableLepton( const reco::GenParticle & daughter );
        bool isElectron( const reco::GenParticle & daughter );
        bool isMuon( const reco::GenParticle & daughter );
        bool isChargedPion( const reco::GenParticle & daughter );
        bool isNeutralPion( const reco::GenParticle & daughter );
        bool isIntermediateResonance( const reco::GenParticle & daughter );
        bool isGamma( const reco::GenParticle & daughter );

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

bool HGCalTriggerNtupleGenTau::isChargedPion( const reco::GenParticle& candidate ){
    bool isChPi=false;
    if( fabs(candidate.pdgId()) == 211 && candidate.status()==1 
        && candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() )
    {
        isChPi=true;
    }
    return isChPi;
}

bool HGCalTriggerNtupleGenTau::isStableLepton( const reco::GenParticle& candidate )
{
    bool isLept=false;
    if( (fabs(candidate.pdgId()) == 11 || fabs(candidate.pdgId()) == 13) && candidate.status()==1 
        && candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() )
    {
        isLept=true;
    }
    return isLept;
}

bool HGCalTriggerNtupleGenTau::isElectron( const reco::GenParticle& candidate )
{
    bool isEle=false;
    if( fabs(candidate.pdgId()) == 11 && candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() )
    {
        isEle=true;
    }
    return isEle;
}

bool HGCalTriggerNtupleGenTau::isMuon( const reco::GenParticle& candidate )
{
    bool isMu=false;
    if( fabs(candidate.pdgId()) == 13 && candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() )
    {
        isMu=true;
    }
    return isMu;
}

bool HGCalTriggerNtupleGenTau::isNeutralPion( const reco::GenParticle& candidate )
{
    bool isPiZero=false;
    if( fabs(candidate.pdgId()) == 111 && candidate.status()==2 && candidate.statusFlags().isTauDecayProduct()
        && !candidate.isDirectPromptTauDecayProductFinalState() )
    {
        isPiZero=true;
    }
    return isPiZero;
}

bool HGCalTriggerNtupleGenTau::isGamma( const reco::GenParticle& candidate )
{
    bool isGammaFromPiZero=false;
    if( fabs(candidate.pdgId()) == 22 && candidate.status()==1 && candidate.statusFlags().isTauDecayProduct() 
        && !candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() )
    {
        isGammaFromPiZero=true;
    }
    return isGammaFromPiZero;
}

bool HGCalTriggerNtupleGenTau::isIntermediateResonance( const reco::GenParticle& candidate )
{
    bool isResonance=false;
    if( ( fabs(candidate.pdgId()) == 213 || fabs(candidate.pdgId()) == 20213 || fabs(candidate.pdgId()) == 24 )
        && candidate.isDirectPromptTauDecayProductFinalState() && candidate.status() == 2  )
    {
        isResonance=true;
    }
    return isResonance;
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
            const edm::RefVector<std::vector<reco::GenParticle> >& daughters = particle.daughterRefVector();

            for( const auto daughter : daughters ){

                std::vector< LorentzVector > finalProd_p4;
                std::vector< int > finalProd_id;

                if( isStableLepton( *daughter ) ){
                    if( isElectron( *daughter ) ){
                        n_ele++;
                    }
                    else if( isMuon(*daughter) ){
                        n_mu++;
                    }
                    finalProd_p4.push_back(daughter->p4());       
                    finalProd_id.push_back(daughter->pdgId());
                    tau_p4vis+=(daughter->p4());
                }        

                /* Here the selection of the decay product according to the Pythia8 decayTree */
                if( isPythia8generator_ ){
                    if( isChargedPion( *daughter ) ){
                        n_pi++;
                        finalProd_p4.push_back(daughter->p4());
                        finalProd_id.push_back(daughter->pdgId());
                        tau_p4vis+=(daughter->p4());
                    }                
                    if( isNeutralPion( *daughter ) ){
                        n_piZero++;
                        const edm::RefVector<std::vector<reco::GenParticle> >& grandaughters = (*daughter).daughterRefVector();
                        for( const auto grandaughter : grandaughters ){
                            if( isGamma( *grandaughter ) ){
                                n_gamma++;
                                finalProd_p4.push_back(grandaughter->p4());
                                finalProd_id.push_back(grandaughter->pdgId());
                                tau_p4vis+=(grandaughter->p4());         
                            }
                        }
                    }
                }

                /* Here the selection of the decay product according to the Pythia6 decayTree */
                else if( !isPythia8generator_ ){            
                    if( isChargedPion( *daughter ) ){
                        n_pi++;
                        finalProd_p4.push_back(daughter->p4());
                        finalProd_id.push_back(daughter->pdgId());
                        tau_p4vis+=(daughter->p4());
                    }
                    if( isNeutralPion( *daughter ) ){
                        n_piZero++;
                        const edm::RefVector<std::vector<reco::GenParticle> >& grandaughters = (*daughter).daughterRefVector();
                        for( const auto grandaughter : grandaughters ){
                            n_gamma++;
                            finalProd_p4.push_back(grandaughter->p4());
                            finalProd_id.push_back(grandaughter->pdgId());
                            tau_p4vis+=(grandaughter->p4());         
                        }          
                    }
                    if( isIntermediateResonance( *daughter ) ){
                        const edm::RefVector<std::vector<reco::GenParticle> >& grandaughters = (*daughter).daughterRefVector();
                        for( const auto grandaughter : grandaughters ){
                            if( isChargedPion( *grandaughter ) ){
                                n_pi++;
                                finalProd_p4.push_back(grandaughter->p4());
                                finalProd_id.push_back(grandaughter->pdgId());
                                tau_p4vis+=(grandaughter->p4());         
                            }                            
                            if( isNeutralPion( *grandaughter ) ){
                                n_piZero++;
                                const edm::RefVector<std::vector<reco::GenParticle> >& descendants = (*grandaughter).daughterRefVector();
                                for( const auto descendant : descendants ){
                                    if( isGamma( *descendant ) ){
                                        n_gamma++;
                                        finalProd_p4.push_back(descendant->p4());
                                        finalProd_id.push_back(descendant->pdgId());
                                        tau_p4vis+=(descendant->p4());         
                                    }
                                }
                            }                            
                        }
                    }
                }

                /* Fill daughter informations */
                for(unsigned j=0; j<finalProd_p4.size(); ++j){
                    tau_products_pt.emplace_back(finalProd_p4.at(j).Pt());
                    tau_products_eta.emplace_back(finalProd_p4.at(j).Eta());
                    tau_products_phi.emplace_back(finalProd_p4.at(j).Phi());
                    tau_products_energy.emplace_back(finalProd_p4.at(j).E());
                    tau_products_mass.emplace_back(finalProd_p4.at(j).M());                                    
                    tau_products_id.emplace_back(finalProd_id.at(j));
                }
                
            } 
           
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

            /* leptonic tau decays */
            if( n_pi == 0 && n_piZero == 0 && n_ele==1 ){ gen_tau_decayMode_.emplace_back(11); }
            else if( n_pi == 0 && n_piZero == 0 && n_mu==1 ){ gen_tau_decayMode_.emplace_back(13); }
            /* 1-prong */
            else if( n_pi == 1 && n_piZero == 0 ){ gen_tau_decayMode_.emplace_back(0); }
            /* 1-prong + pi0s */            
            else if( n_pi == 1 && n_piZero >= 1 ){ gen_tau_decayMode_.emplace_back(1); }
            /* 3-prongs */
            else if( n_pi == 3 && n_piZero == 0 ){ gen_tau_decayMode_.emplace_back(4); }
            /* 3-prongs + pi0s */
            else if( n_pi == 3 && n_piZero >= 1 ){ gen_tau_decayMode_.emplace_back(5); }
            /* other decays */
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




