#include "GeneratorInterface/Core/interface/EmbeddingHepMCFilter.h"

#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/trim_all.hpp"

EmbeddingHepMCFilter::EmbeddingHepMCFilter(const edm::ParameterSet &iConfig)
    : ZPDGID_(iConfig.getParameter<int>("BosonPDGID")),
      includeDY_(iConfig.existsAs<bool>("IncludeDY") ? iConfig.getParameter<bool>("IncludeDY") : false) {
  // Defining standard decay channels
  ee.fill(TauDecayMode::Electron);
  ee.fill(TauDecayMode::Electron);
  mm.fill(TauDecayMode::Muon);
  mm.fill(TauDecayMode::Muon);
  hh.fill(TauDecayMode::Hadronic);
  hh.fill(TauDecayMode::Hadronic);
  em.fill(TauDecayMode::Electron);
  em.fill(TauDecayMode::Muon);
  eh.fill(TauDecayMode::Electron);
  eh.fill(TauDecayMode::Hadronic);
  mh.fill(TauDecayMode::Muon);
  mh.fill(TauDecayMode::Hadronic);

  // Filling CutContainers

  std::string cut_string_elel = iConfig.getParameter<std::string>("ElElCut");
  std::string cut_string_mumu = iConfig.getParameter<std::string>("MuMuCut");
  std::string cut_string_hadhad = iConfig.getParameter<std::string>("HadHadCut");
  std::string cut_string_elmu = iConfig.getParameter<std::string>("ElMuCut");
  std::string cut_string_elhad = iConfig.getParameter<std::string>("ElHadCut");
  std::string cut_string_muhad = iConfig.getParameter<std::string>("MuHadCut");

  std::vector<std::string> use_final_states = iConfig.getParameter<std::vector<std::string> >("Final_States");

  for (std::vector<std::string>::const_iterator final_state = use_final_states.begin();
       final_state != use_final_states.end();
       ++final_state) {
    if ((*final_state) == "ElEl")
      fill_cuts(cut_string_elel, ee);
    else if ((*final_state) == "MuMu")
      fill_cuts(cut_string_mumu, mm);
    else if ((*final_state) == "HadHad")
      fill_cuts(cut_string_hadhad, hh);
    else if ((*final_state) == "ElMu")
      fill_cuts(cut_string_elmu, em);
    else if ((*final_state) == "ElHad")
      fill_cuts(cut_string_elhad, eh);
    else if ((*final_state) == "MuHad")
      fill_cuts(cut_string_muhad, mh);
    else
      edm::LogWarning("EmbeddingHepMCFilter")
          << (*final_state)
          << " this decay channel is not supported. Please choose on of (ElEl,MuMu,HadHad,ElMu,ElHad,MuHad)";
  }
}

EmbeddingHepMCFilter::~EmbeddingHepMCFilter() {}

bool EmbeddingHepMCFilter::filter(const HepMC::GenEvent *evt) {
  //Reset DecayChannel_ and p4VisPair_ at the beginning of each event.
  DecayChannel_.reset();
  std::vector<reco::Candidate::LorentzVector> p4VisPair_;

  // Going through the particle list. Mother particles are allways before their children.
  // One can stop the loop after the second tau is reached and processed.
  for (HepMC::GenEvent::particle_const_iterator particle = evt->particles_begin(); particle != evt->particles_end();
       ++particle) {
    int mom_id = 0;           // no particle available with PDG ID 0
    bool isHardProc = false;  // mother is ZPDGID_, or is lepton from hard process (DY process qq -> ll)
    int pdg_id = std::abs((*particle)->pdg_id());
    HepMC::GenVertex *vertex = (*particle)->production_vertex();
    if (vertex != nullptr) {  // search for the mom via the production_vertex
      HepMC::GenVertex::particles_in_const_iterator mom = vertex->particles_in_const_begin();
      if (mom != vertex->particles_in_const_end()) {
        mom_id = std::abs((*mom)->pdg_id());  // mom was found
      }
      if (mom_id == ZPDGID_) {
        isHardProc = true;  // intermediate boson
      } else if (includeDY_ && 11 <= pdg_id && pdg_id <= 16 && mcTruthHelper_.isFirstCopy(**particle) &&
                 mcTruthHelper_.fromHardProcess(**particle)) {
        edm::LogInfo("EmbeddingHepMCFilter") << (*particle)->pdg_id() << " with mother " << (*mom)->pdg_id();
        isHardProc = true;  // assume Drell-Yan qq -> ll without intermediate boson
      }
    }

    if (!isHardProc) {
      continue;
    } else if (pdg_id == tauonPDGID_) {
      reco::Candidate::LorentzVector p4Vis;
      decay_and_sump4Vis((*particle), p4Vis);  // recursive access to final states.
      p4VisPair_.push_back(p4Vis);
    } else if (pdg_id == muonPDGID_) {  // Also handle the option when Z-> mumu
      reco::Candidate::LorentzVector p4Vis = (reco::Candidate::LorentzVector)(*particle)->momentum();
      DecayChannel_.fill(TauDecayMode::Muon);  // take the muon cuts
      p4VisPair_.push_back(p4Vis);
    } else if (pdg_id == electronPDGID_) {  // Also handle the option when Z-> ee
      reco::Candidate::LorentzVector p4Vis = (reco::Candidate::LorentzVector)(*particle)->momentum();
      DecayChannel_.fill(TauDecayMode::Electron);  // take the electron cuts
      p4VisPair_.push_back(p4Vis);
    }
  }
  // Putting DecayChannel_ in default convention:
  // For mixed decay channels use the Electron_Muon, Electron_Hadronic, Muon_Hadronic convention.
  // For symmetric decay channels (e.g. Muon_Muon) use Leading_Trailing convention with respect to Pt.
  if (p4VisPair_.size() == 2) {
    sort_by_convention(p4VisPair_);
    edm::LogInfo("EmbeddingHepMCFilter") << "Quantities of the visible decay products:"
                                         << "\tPt's: "
                                         << " 1st " << p4VisPair_[0].Pt() << ", 2nd " << p4VisPair_[1].Pt()
                                         << "\tEta's: "
                                         << " 1st " << p4VisPair_[0].Eta() << ", 2nd " << p4VisPair_[1].Eta()
                                         << " decay channel: " << return_mode(DecayChannel_.first)
                                         << return_mode(DecayChannel_.second);
  } else {
    edm::LogError("EmbeddingHepMCFilter") << "Size less non equal two";
  }

  return apply_cuts(p4VisPair_);
}

void EmbeddingHepMCFilter::decay_and_sump4Vis(HepMC::GenParticle *particle, reco::Candidate::LorentzVector &p4Vis) {
  bool decaymode_known = false;
  for (HepMC::GenVertex::particle_iterator daughter = particle->end_vertex()->particles_begin(HepMC::children);
       daughter != particle->end_vertex()->particles_end(HepMC::children);
       ++daughter) {
    bool neutrino = (std::abs((*daughter)->pdg_id()) == tauon_neutrino_PDGID_) ||
                    (std::abs((*daughter)->pdg_id()) == muon_neutrino_PDGID_) ||
                    (std::abs((*daughter)->pdg_id()) == electron_neutrino_PDGID_);

    // Determining DecayMode, if particle is tau lepton.
    // Asuming, that there are only the two tauons from the Z-boson.
    // This is the case for the simulated Z->tautau event constructed by EmbeddingLHEProducer.
    if (std::abs(particle->pdg_id()) == tauonPDGID_ && !decaymode_known) {
      // use these bools to protect againt taus that aren't the last copy (which "decay" to a tau and a gamma)
      bool isGamma = std::abs((*daughter)->pdg_id()) == 22;
      bool isTau = std::abs((*daughter)->pdg_id()) == 15;
      if (std::abs((*daughter)->pdg_id()) == muonPDGID_) {
        DecayChannel_.fill(TauDecayMode::Muon);
        decaymode_known = true;
      } else if (std::abs((*daughter)->pdg_id()) == electronPDGID_) {
        DecayChannel_.fill(TauDecayMode::Electron);
        decaymode_known = true;
      } else if (!neutrino && !isGamma && !isTau) {
        DecayChannel_.fill(TauDecayMode::Hadronic);
        decaymode_known = true;
      }
    }
    // Adding up all visible momentum in recursive way.
    if ((*daughter)->status() == 1 && !neutrino)
      p4Vis += (reco::Candidate::LorentzVector)(*daughter)->momentum();
    else if (!neutrino)
      decay_and_sump4Vis((*daughter), p4Vis);
  }
}

void EmbeddingHepMCFilter::sort_by_convention(std::vector<reco::Candidate::LorentzVector> &p4VisPair) {
  bool mixed_false_order =
      (DecayChannel_.first == TauDecayMode::Hadronic && DecayChannel_.second == TauDecayMode::Muon) ||
      (DecayChannel_.first == TauDecayMode::Hadronic && DecayChannel_.second == TauDecayMode::Electron) ||
      (DecayChannel_.first == TauDecayMode::Muon && DecayChannel_.second == TauDecayMode::Electron);

  if (DecayChannel_.first == DecayChannel_.second && p4VisPair[0].Pt() < p4VisPair[1].Pt()) {
    edm::LogVerbatim("EmbeddingHepMCFilter") << "Changing symmetric channels to Leading_Trailing convention in Pt";
    edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's before: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
    std::reverse(p4VisPair.begin(), p4VisPair.end());
    edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's after: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
  } else if (mixed_false_order) {
    edm::LogVerbatim("EmbeddingHepMCFilter") << "Swapping order of mixed channels";
    edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's before: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
    DecayChannel_.reverse();
    edm::LogVerbatim("EmbeddingHepMCFilter")
        << "DecayChannel: " << return_mode(DecayChannel_.first) << return_mode(DecayChannel_.second);
    std::reverse(p4VisPair.begin(), p4VisPair.end());
    edm::LogVerbatim("EmbeddingHepMCFilter") << "Pt's after: " << p4VisPair[0].Pt() << " " << p4VisPair[1].Pt();
  }
}

bool EmbeddingHepMCFilter::apply_cuts(std::vector<reco::Candidate::LorentzVector> &p4VisPair) {
  for (std::vector<CutsContainer>::const_iterator cut = cuts_.begin(); cut != cuts_.end(); ++cut) {
    if (DecayChannel_.first == cut->decaychannel.first &&
        DecayChannel_.second == cut->decaychannel.second) {  // First the match to the decay channel
      edm::LogInfo("EmbeddingHepMCFilter")
          << "Cut pt1 = " << cut->pt1 << " pt2 = " << cut->pt2 << " abs(eta1) = " << cut->eta1
          << " abs(eta2) = " << cut->eta2 << " decay channel: " << return_mode(cut->decaychannel.first)
          << return_mode(cut->decaychannel.second);

      if ((cut->pt1 == -1. || (p4VisPair[0].Pt() > cut->pt1)) && (cut->pt2 == -1. || (p4VisPair[1].Pt() > cut->pt2)) &&
          (cut->eta1 == -1. || (std::abs(p4VisPair[0].Eta()) < cut->eta1)) &&
          (cut->eta2 == -1. || (std::abs(p4VisPair[1].Eta()) < cut->eta2))) {
        edm::LogInfo("EmbeddingHepMCFilter") << "This cut was passed (Stop here and take the event)";
        return true;
      }
    }
  }
  return false;
}

void EmbeddingHepMCFilter::fill_cuts(std::string cut_string, EmbeddingHepMCFilter::DecayChannel &dc) {
  edm::LogInfo("EmbeddingHepMCFilter") << return_mode(dc.first) << return_mode(dc.second) << "Cut : " << cut_string;
  boost::trim_fill(cut_string, "");
  std::vector<std::string> cut_paths;
  boost::split(cut_paths, cut_string, boost::is_any_of("||"), boost::token_compress_on);
  for (unsigned int i = 0; i < cut_paths.size(); ++i) {
    // Translating the cuts of a path into a struct which is later accessed to apply them on a event.
    CutsContainer cut;
    fill_cut(cut_paths[i], dc, cut);
    cuts_.push_back(cut);
  }
}

void EmbeddingHepMCFilter::fill_cut(std::string cut_string,
                                    EmbeddingHepMCFilter::DecayChannel &dc,
                                    CutsContainer &cut) {
  cut.decaychannel = dc;

  boost::replace_all(cut_string, "(", "");
  boost::replace_all(cut_string, ")", "");
  std::vector<std::string> single_cuts;
  boost::split(single_cuts, cut_string, boost::is_any_of("&&"), boost::token_compress_on);
  for (unsigned int i = 0; i < single_cuts.size(); ++i) {
    std::string pt1_str, pt2_str, eta1_str, eta2_str;
    if (dc.first == dc.second) {
      pt1_str = return_mode(dc.first) + "1" + ".Pt" + ">";
      pt2_str = return_mode(dc.second) + "2" + ".Pt" + ">";
      eta1_str = return_mode(dc.first) + "1" + ".Eta" + "<";
      eta2_str = return_mode(dc.second) + "2" + ".Eta" + "<";
    } else {
      pt1_str = return_mode(dc.first) + ".Pt" + ">";
      pt2_str = return_mode(dc.second) + ".Pt" + ">";
      eta1_str = return_mode(dc.first) + ".Eta" + "<";
      eta2_str = return_mode(dc.second) + ".Eta" + "<";
    }

    if (boost::find_first(single_cuts[i], pt1_str)) {
      boost::erase_first(single_cuts[i], pt1_str);
      cut.pt1 = std::stod(single_cuts[i]);
    } else if (boost::find_first(single_cuts[i], pt2_str)) {
      boost::erase_first(single_cuts[i], pt2_str);
      cut.pt2 = std::stod(single_cuts[i]);
    } else if (boost::find_first(single_cuts[i], eta1_str)) {
      boost::erase_first(single_cuts[i], eta1_str);
      cut.eta1 = std::stod(single_cuts[i]);
    } else if (boost::find_first(single_cuts[i], eta2_str)) {
      boost::erase_first(single_cuts[i], eta2_str);
      cut.eta2 = std::stod(single_cuts[i]);
    }
  }
}
