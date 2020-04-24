#ifndef GeneratorInterface_RivetInterface_RivetAnalysis_H
#define GeneratorInterface_RivetInterface_RivetAnalysis_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Particle.hh"
#include "Rivet/Particle.fhh"
#include "Rivet/Event.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/JetAlg.hh"
#include "Rivet/Projections/ChargedLeptons.hh"
#include "Rivet/Projections/PromptFinalState.hh"
#include "Rivet/Projections/DressedLeptons.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Projections/MissingMomentum.hh"
#include "Rivet/Tools/RivetHepMC.hh"

namespace Rivet {

  class RivetAnalysis : public Analysis {

    public:
      std::vector<DressedLepton> leptons() const {return _leptons;}
      ParticleVector photons() const {return _photons;}
      ParticleVector neutrinos() const {return _neutrinos;}
      Jets jets() const {return _jets;}
      Jets fatjets() const {return _fatjets;}
      Vector3 met() const {return _met;}

    private:
      bool _usePromptFinalStates;
      bool _excludePromptLeptonsFromJetClustering;
      bool _excludeNeutrinosFromJetClustering;
      
      double _particleMinPt, _particleMaxEta;
      double _lepConeSize, _lepMinPt, _lepMaxEta;
      double _jetConeSize, _jetMinPt, _jetMaxEta;
      double _fatJetConeSize, _fatJetMinPt, _fatJetMaxEta;
      
      std::vector<DressedLepton> _leptons;
      ParticleVector _photons, _neutrinos;
      Jets _jets, _fatjets;
      Vector3 _met;

    public:
      RivetAnalysis(const edm::ParameterSet& pset) : Analysis("RivetAnalysis"),
      _usePromptFinalStates(pset.getParameter<bool>("usePromptFinalStates")),
      _excludePromptLeptonsFromJetClustering(pset.getParameter<bool>("excludePromptLeptonsFromJetClustering")),
      _excludeNeutrinosFromJetClustering(pset.getParameter<bool>("excludeNeutrinosFromJetClustering")),

      _particleMinPt  (pset.getParameter<double>("particleMinPt")),
      _particleMaxEta (pset.getParameter<double>("particleMaxEta")),
      
      _lepConeSize (pset.getParameter<double>("lepConeSize")),
      _lepMinPt    (pset.getParameter<double>("lepMinPt")),
      _lepMaxEta   (pset.getParameter<double>("lepMaxEta")),
      
      _jetConeSize (pset.getParameter<double>("jetConeSize")),
      _jetMinPt    (pset.getParameter<double>("jetMinPt")),
      _jetMaxEta   (pset.getParameter<double>("jetMaxEta")),
      
      _fatJetConeSize (pset.getParameter<double>("fatJetConeSize")),
      _fatJetMinPt    (pset.getParameter<double>("fatJetMinPt")),
      _fatJetMaxEta   (pset.getParameter<double>("fatJetMaxEta"))
      {
      }

      // Initialize Rivet projections
      void init() {
        // Cuts
        Cut particle_cut = (Cuts::abseta < _particleMaxEta) and (Cuts::pT > _particleMinPt*GeV);
        Cut lepton_cut   = (Cuts::abseta < _lepMaxEta)      and (Cuts::pT > _lepMinPt*GeV);
        
        // Generic final state
        FinalState fs(particle_cut);
        
        // Dressed leptons
        ChargedLeptons charged_leptons(fs);
        IdentifiedFinalState photons(fs);
        photons.acceptIdPair(PID::PHOTON);
        
        PromptFinalState prompt_leptons(charged_leptons);
        prompt_leptons.acceptMuonDecays(true);
        prompt_leptons.acceptTauDecays(true);
        
        PromptFinalState prompt_photons(photons);
        prompt_photons.acceptMuonDecays(true);
        prompt_photons.acceptTauDecays(true);
        
        // useDecayPhotons=true allows for photons with tau ancestor,
        // photons from hadrons are vetoed by the PromptFinalState;
        // will be default DressedLeptons behaviour for Rivet >= 2.5.4
        DressedLeptons dressed_leptons(prompt_photons, prompt_leptons, _lepConeSize, 
                       lepton_cut, /*cluster*/ true, /*useDecayPhotons*/ true);
        if (not _usePromptFinalStates)
          dressed_leptons = DressedLeptons(photons, charged_leptons, _lepConeSize, 
                            lepton_cut, /*cluster*/ true, /*useDecayPhotons*/ true);
        addProjection(dressed_leptons, "DressedLeptons");
        
        // Photons
        if (_usePromptFinalStates) {
          // We remove the photons used up for lepton dressing in this case
          VetoedFinalState vetoed_prompt_photons(prompt_photons);
          vetoed_prompt_photons.addVetoOnThisFinalState(dressed_leptons);
          addProjection(vetoed_prompt_photons, "Photons");
        }
        else
          addProjection(photons, "Photons");
        
        // Jets
        VetoedFinalState fsForJets(fs);
        if (_usePromptFinalStates and _excludePromptLeptonsFromJetClustering)
          fsForJets.addVetoOnThisFinalState(dressed_leptons);
        JetAlg::InvisiblesStrategy invisiblesStrategy = JetAlg::DECAY_INVISIBLES;
        if (_excludeNeutrinosFromJetClustering)
          invisiblesStrategy = JetAlg::NO_INVISIBLES;
        addProjection(FastJets(fsForJets, FastJets::ANTIKT, _jetConeSize,
                               JetAlg::ALL_MUONS, invisiblesStrategy), "Jets");
        
        // FatJets
        addProjection(FastJets(fsForJets, FastJets::ANTIKT, _fatJetConeSize), "FatJets");
        
        // Neutrinos
        IdentifiedFinalState neutrinos(fs);
        neutrinos.acceptNeutrinos();
        if (_usePromptFinalStates) {
          PromptFinalState prompt_neutrinos(neutrinos);
          prompt_neutrinos.acceptMuonDecays(true);
          prompt_neutrinos.acceptTauDecays(true);
          addProjection(prompt_neutrinos, "Neutrinos");
        }
        else
          addProjection(neutrinos, "Neutrinos");
        
        // MET
        addProjection(MissingMomentum(fs), "MET");
      };

      // Apply Rivet projections
      void analyze(const Event& event) {
        _jets.clear();
        _fatjets.clear();
        _leptons.clear();
        _photons.clear();
        _neutrinos.clear();
        
        // Get analysis objects from projections
        Cut jet_cut    = (Cuts::abseta < _jetMaxEta)    and (Cuts::pT > _jetMinPt*GeV);
        Cut fatjet_cut = (Cuts::abseta < _fatJetMaxEta) and (Cuts::pT > _fatJetMinPt*GeV);
        
        _leptons   = applyProjection<DressedLeptons>(event, "DressedLeptons").dressedLeptons();
        _jets      = applyProjection<FastJets>(event, "Jets").jetsByPt(jet_cut);
        _fatjets   = applyProjection<FastJets>(event, "FatJets").jetsByPt(fatjet_cut);
        _photons   = applyProjection<FinalState>(event, "Photons").particlesByPt();
        _neutrinos = applyProjection<FinalState>(event, "Neutrinos").particlesByPt();
        _met       = applyProjection<MissingMomentum>(event, "MET").missingMomentum().p3();
      };

      // Do nothing here
      void finalize() {};

  };

}

#endif
