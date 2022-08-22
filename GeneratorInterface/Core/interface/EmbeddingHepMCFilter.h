#ifndef __EMBEDDINGHEPMCFILTER__
#define __EMBEDDINGHEPMCFILTER__

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class EmbeddingHepMCFilter : public BaseHepMCFilter {
private:
  const int tauon_neutrino_PDGID_ = 16;
  const int tauonPDGID_ = 15;
  const int muon_neutrino_PDGID_ = 14;
  const int muonPDGID_ = 13;
  const int electron_neutrino_PDGID_ = 12;
  const int electronPDGID_ = 11;
  int ZPDGID_ = 23;
  bool includeDY_ = false;
  MCTruthHelper<HepMC::GenParticle> mcTruthHelper_;

  enum class TauDecayMode : int { Unfilled = -1, Muon = 0, Electron = 1, Hadronic = 2 };

  std::string return_mode(TauDecayMode mode) {
    if (mode == TauDecayMode::Muon)
      return "Mu";
    else if (mode == TauDecayMode::Electron)
      return "El";
    else if (mode == TauDecayMode::Hadronic)
      return "Had";
    else
      return "Undefined";
  }

  struct DecayChannel {
    TauDecayMode first = TauDecayMode::Unfilled;
    TauDecayMode second = TauDecayMode::Unfilled;

    void fill(TauDecayMode mode) {
      if (first == TauDecayMode::Unfilled)
        first = mode;
      else if (second == TauDecayMode::Unfilled)
        second = mode;
    };
    void reset() {
      first = TauDecayMode::Unfilled;
      second = TauDecayMode::Unfilled;
    }
    void reverse() {
      TauDecayMode tmp = first;
      first = second;
      second = tmp;
    }
  };

  DecayChannel ee, mm, hh, em, eh, mh;

  struct CutsContainer {
    double pt1 = -1.;
    double pt2 = -1.;
    double eta1 = -1.;  // since we use abs eta values the -1 as default is OK
    double eta2 = -1.;
    DecayChannel decaychannel;
  };

  std::vector<CutsContainer> cuts_;
  DecayChannel DecayChannel_;

  virtual void fill_cut(std::string cut_string, EmbeddingHepMCFilter::DecayChannel &dc, CutsContainer &cut);
  virtual void fill_cuts(std::string cut_string, EmbeddingHepMCFilter::DecayChannel &dc);

  virtual void decay_and_sump4Vis(HepMC::GenParticle *particle, reco::Candidate::LorentzVector &p4Vis);
  virtual void sort_by_convention(std::vector<reco::Candidate::LorentzVector> &p4VisPair);
  virtual bool apply_cuts(std::vector<reco::Candidate::LorentzVector> &p4VisPair);

public:
  explicit EmbeddingHepMCFilter(const edm::ParameterSet &);
  ~EmbeddingHepMCFilter() override;

  bool filter(const HepMC::GenEvent *evt) override;
};

#endif
