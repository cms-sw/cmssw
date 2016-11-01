#include "GeneratorInterface/Core/interface/EmbeddingHepMCFilter.h"


EmbeddingHepMCFilter::EmbeddingHepMCFilter(const edm::ParameterSet & iConfig)
{    
    // Defining standard decay channels
    ee.fill(TauDecayMode::Electron); ee.fill(TauDecayMode::Electron);
    mm.fill(TauDecayMode::Muon); mm.fill(TauDecayMode::Muon);
    hh.fill(TauDecayMode::Hadronic); hh.fill(TauDecayMode::Hadronic);
    em.fill(TauDecayMode::Electron); em.fill(TauDecayMode::Muon);
    eh.fill(TauDecayMode::Electron); eh.fill(TauDecayMode::Hadronic);
    mh.fill(TauDecayMode::Muon); mh.fill(TauDecayMode::Hadronic);
    
    // Filling CutContainers
    for (auto & dc_m : decaychannel_markers)
    {
        // Reading out the cut string from the config
        std::string cut_string_copy = iConfig.getUntrackedParameter<std::string>(dc_m.first + "Cut", "REJECT");
        edm::LogInfo("EmbeddingHepMCFilter") << dc_m.first << " : " << cut_string_copy;
        boost::trim_fill(cut_string_copy, "");
        
        // Splitting cut string by paths
        std::vector<std::string> cut_paths;
        boost::split(cut_paths, cut_string_copy, boost::is_any_of("||"), boost::token_compress_on);
        for(unsigned int i=0; i<cut_paths.size(); ++i)
        {
            // Translating the cuts of a path into a struct which is later accessed to apply them on a event.
            CutsContainer cut;
            if (cut_paths[i] != "REJECT")
            {
                fill_cut(cut_paths[i], dc_m.second, cut);
                cuts_.push_back(cut);
            }
        }
    }
}

EmbeddingHepMCFilter::~EmbeddingHepMCFilter()
{
}


bool
EmbeddingHepMCFilter::filter(const HepMC::GenEvent* evt)
{
    //Reset DecayChannel_ and p4VisPair_ at the beginning of each event.
    DecayChannel_.reset();
    std::vector<reco::Candidate::LorentzVector> p4VisPair_;
    
    // Going through the particle list. Mother particles are allways before their children. 
    // One can stop the loop after the second tau is reached and processed.
    for ( HepMC::GenEvent::particle_const_iterator particle = evt->particles_begin(); particle != evt->particles_end(); ++particle )
    {
       // (*particle)->print();
	int mom_id = 0; // No particle available with PDG ID 0
	if ((*particle)->production_vertex() != 0) 
	  if ((*particle)->production_vertex()->particles_in_const_begin() != (*particle)->production_vertex()->particles_in_const_end())  mom_id =  (*(*particle)->production_vertex()->particles_in_const_begin())->pdg_id();
        if (std::abs((*particle)->pdg_id()) == tauonPDGID_  && mom_id == ZPDGID_)
	{
          reco::Candidate::LorentzVector p4Vis;
          decay_and_sump4Vis((*particle), p4Vis); // recursive access to final states.
          p4VisPair_.push_back(p4Vis);	  
        }
         if (std::abs((*particle)->pdg_id()) == muonPDGID_  && mom_id == ZPDGID_) // Also handle the option when Z-> mumu
	{
          reco::Candidate::LorentzVector p4Vis = (reco::Candidate::LorentzVector) (*particle)->momentum();
	  DecayChannel_.fill(TauDecayMode::Muon); // treat it like muon decays
          p4VisPair_.push_back(p4Vis);
	}

    }
    // Putting DecayChannel_ in default convention:
    // For mixed decay channels use the Electron_Muon, Electron_Hadronic, Muon_Hadronic convention.
    // For symmetric decay channels (e.g. Muon_Muon) use Leading_Trailing convention with respect to Pt.
    sort_by_convention(DecayChannel_, p4VisPair_);
    edm::LogInfo("EmbeddingHepMCFilter") << "Quantities of the visible decay products:";
    edm::LogInfo("EmbeddingHepMCFilter") << "Pt's: " << " 1st " << p4VisPair_[0].Pt() << ", 2nd " << p4VisPair_[1].Pt();
    edm::LogInfo("EmbeddingHepMCFilter") << "Eta's: " << " 1st " << p4VisPair_[0].Eta() << ", 2nd " << p4VisPair_[1].Eta();
    
    return apply_cuts(DecayChannel_, p4VisPair_, cuts_);
}


void
EmbeddingHepMCFilter::decay_and_sump4Vis(HepMC::GenParticle* particle, reco::Candidate::LorentzVector &p4Vis)
{
    bool decaymode_known = false;
    for (HepMC::GenVertex::particle_iterator daughter = particle->end_vertex()->particles_begin(HepMC::children); 
    daughter != particle->end_vertex()->particles_end(HepMC::children); ++daughter)
    {
        bool neutrino = (std::abs((*daughter)->pdg_id()) == tauon_neutrino_PDGID_) ||
                        (std::abs((*daughter)->pdg_id()) == muon_neutrino_PDGID_) ||
                        (std::abs((*daughter)->pdg_id()) == electron_neutrino_PDGID_);
        
        // Determining DecayMode, if particle is tau lepton.
        // Asuming, that there are only the two tauons from the Z-boson.
        // This is the case for the simulated Z->tautau event constructed by EmbeddingLHEProducer. 
        if (std::abs(particle->pdg_id()) == tauonPDGID_ && !decaymode_known)
        {
            if (std::abs((*daughter)->pdg_id()) == muonPDGID_)
            {
                DecayChannel_.fill(TauDecayMode::Muon);
                decaymode_known = true;
            }
            else if (std::abs((*daughter)->pdg_id()) == electronPDGID_)
            {
                DecayChannel_.fill(TauDecayMode::Electron);
                decaymode_known = true;
            }
            else if (!neutrino)
            {
                DecayChannel_.fill(TauDecayMode::Hadronic);
                decaymode_known = true;
            }
        }
        // Adding up all visible momentum in recursive way.
        if ((*daughter)->status() == 1 && !neutrino) p4Vis += (reco::Candidate::LorentzVector) (*daughter)->momentum();
        else if (!neutrino) decay_and_sump4Vis((*daughter), p4Vis);
    }
}


void
EmbeddingHepMCFilter::sort_by_convention(DecayChannel &dc, std::vector<reco::Candidate::LorentzVector> &p4VisPair)
{
    bool mixed_false_order = (dc.first == TauDecayMode::Hadronic && dc.second == TauDecayMode::Muon) ||
                             (dc.first == TauDecayMode::Hadronic && dc.second == TauDecayMode::Electron) ||
                             (dc.first == TauDecayMode::Muon && dc.second == TauDecayMode::Electron);
    
    if (dc.first == dc.second && p4VisPair[0].Pt() < p4VisPair[1].Pt())
    {
        edm::LogVerbatim("EmbeddingHepMCFilter") << "Changing symmetric channels to Leading_Trailing convention in Pt";
        edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's before: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
        std::reverse(p4VisPair.begin(),p4VisPair.end());
        edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's after: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
    }
    else if (mixed_false_order)
    {
        edm::LogVerbatim("EmbeddingHepMCFilter") << "Swapping order of mixed channels";
        edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's before: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
        dc.reverse();
        edm::LogVerbatim("EmbeddingHepMCFilter") << "DecayChannel: " << return_mode(dc.first) << return_mode(dc.second);
        std::reverse(p4VisPair.begin(),p4VisPair.end());
        edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's after: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
    }
}

bool
EmbeddingHepMCFilter::apply_cuts(DecayChannel &dc, std::vector<reco::Candidate::LorentzVector> &p4VisPair, std::vector<CutsContainer> &cuts)
{
    for (unsigned int i=0; i<cuts.size(); ++i)
    {

        bool all_cuts_passed = false;
        if(dc.first == cuts[i].decaychannel.first && dc.second ==  cuts[i].decaychannel.second)
        {
            edm::LogInfo("EmbeddingHepMCFilter") << "Cut number " << i << " pt1 = " << cuts[i].pt1 << " pt2 = " << cuts[i].pt2
            << " abs(eta1) = " << cuts[i].eta1 << " abs(eta2) = " << cuts[i].eta2
            << " decay channel: " << return_mode(cuts[i].decaychannel.first)
            << return_mode(cuts[i].decaychannel.second);
            
            if(cuts[i].pt1 != -1. && !(p4VisPair[0].Pt() > cuts[i].pt1)) all_cuts_passed = false;
            else if (cuts[i].pt2 != -1. && !(p4VisPair[1].Pt() > cuts[i].pt2)) all_cuts_passed = false;
            else if (cuts[i].eta1 != -1. && !(std::abs(p4VisPair[0].Eta()) < cuts[i].eta1)) all_cuts_passed = false;
            else if (cuts[i].eta2 != -1. && !(std::abs(p4VisPair[1].Eta()) < cuts[i].eta2)) all_cuts_passed = false;
            else all_cuts_passed = true;
        }
        if (all_cuts_passed)
        {
            edm::LogInfo("EmbeddingHepMCFilter") << "All cuts of one path passed!!!!";
            return true;
        }
    }
    return false;
}

void 
EmbeddingHepMCFilter::fill_cut(std::string cut_string, EmbeddingHepMCFilter::DecayChannel &dc, CutsContainer &cut)
{
            cut.decaychannel = dc;
            
            boost::replace_all(cut_string,"(","");
            boost::replace_all(cut_string,")","");
            std::vector<std::string> single_cuts;
            boost::split(single_cuts, cut_string, boost::is_any_of("&&"), boost::token_compress_on);
            for (unsigned int i=0; i<single_cuts.size(); ++i)
            {
                std::string pt1_str, pt2_str, eta1_str, eta2_str;
                if (dc.first == dc.second)
                {
                    pt1_str = return_mode(dc.first)+"1"+".Pt"+">";
                    pt2_str = return_mode(dc.second)+"2"+".Pt"+">";
                    eta1_str = return_mode(dc.first)+"1"+".Eta"+"<";
                    eta2_str = return_mode(dc.second)+"2"+".Eta"+"<";
                }
                else
                {
                    pt1_str = return_mode(dc.first)+".Pt"+">";
                    pt2_str = return_mode(dc.second)+".Pt"+">";
                    eta1_str = return_mode(dc.first)+".Eta"+"<";
                    eta2_str = return_mode(dc.second)+".Eta"+"<";
                }
                
                if(boost::find_first(single_cuts[i], pt1_str))
                {
                    boost::erase_first(single_cuts[i], pt1_str);
                    cut.pt1 = std::stod(single_cuts[i]);
                }
                else if (boost::find_first(single_cuts[i], pt2_str))
                {
                    boost::erase_first(single_cuts[i], pt2_str);
                    cut.pt2 = std::stod(single_cuts[i]);
                }
                else if (boost::find_first(single_cuts[i], eta1_str))
                {
                    boost::erase_first(single_cuts[i], eta1_str);
                    cut.eta1 = std::stod(single_cuts[i]);
                }
                else if (boost::find_first(single_cuts[i], eta2_str))
                {
                    boost::erase_first(single_cuts[i], eta2_str);
                    cut.eta2 = std::stod(single_cuts[i]);
                }
            }
}