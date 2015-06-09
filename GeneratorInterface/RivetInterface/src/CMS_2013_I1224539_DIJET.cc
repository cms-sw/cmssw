// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/WFinder.hh"
#include "Rivet/Projections/ZFinder.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"

namespace Rivet {


  class CMS_2013_I1224539_DIJET : public Analysis {
  public:

    /// @name Constructors etc.
    //@{

    /// Constructor
    CMS_2013_I1224539_DIJET()
      : Analysis("CMS_2013_I1224539_DIJET"),
        _filter(fastjet::Filter(fastjet::JetDefinition(fastjet::cambridge_algorithm, 0.3), fastjet::SelectorNHardest(3))),
        _trimmer(fastjet::Filter(fastjet::JetDefinition(fastjet::kt_algorithm, 0.2), fastjet::SelectorPtFractionMin(0.03))),
        _pruner(fastjet::Pruner(fastjet::cambridge_algorithm, 0.1, 0.5))
    {    }

    //@}


  public:

    /// @name Analysis methods
    //@{

    /// Book histograms and initialise projections before the run
    void init() {
      FinalState fs(-2.4, 2.4, 0*GeV);
      addProjection(fs, "FS");

      // Jet collections
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.7), "JetsAK7");
      addProjection(FastJets(fs, FastJets::CAM, 0.8), "JetsCA8");
      addProjection(FastJets(fs, FastJets::CAM, 1.2), "JetsCA12");

      // Histograms
      for (size_t i = 0; i < N_PT_BINS_dj; ++i ) {
        _h_ungroomedAvgJetMass_dj[i] = bookHisto1D(i+1+0*N_PT_BINS_dj, 1, 1);
        _h_filteredAvgJetMass_dj[i] = bookHisto1D(i+1+1*N_PT_BINS_dj, 1, 1);
        _h_trimmedAvgJetMass_dj[i] = bookHisto1D(i+1+2*N_PT_BINS_dj, 1, 1);
        _h_prunedAvgJetMass_dj[i] = bookHisto1D(i+1+3*N_PT_BINS_dj, 1, 1);
      }
    }


    // Find the pT histogram bin index for value pt (in GeV), to hack a 2D histogram equivalent
    /// @todo Use a YODA axis/finder alg when available
    size_t findPtBin(double ptJ) {
      const double ptBins_dj[N_PT_BINS_dj+1] = { 220.0, 300.0, 450.0, 500.0, 600.0, 800.0, 1000.0, 1500.0};
      for (size_t ibin = 0; ibin < N_PT_BINS_dj; ++ibin) {
        if (inRange(ptJ, ptBins_dj[ibin], ptBins_dj[ibin+1])) return ibin;
      }
      return N_PT_BINS_dj;
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();

      // Look at events with >= 2 jets
      const PseudoJets& psjetsAK7 = applyProjection<FastJets>(event, "JetsAK7").pseudoJetsByPt( 50.0*GeV );
      if (psjetsAK7.size() < 2) vetoEvent;

      // Get the leading two jets and find their average pT
      const fastjet::PseudoJet& j0 = psjetsAK7[0];
      const fastjet::PseudoJet& j1 = psjetsAK7[1];
      double ptAvg = 0.5 * (j0.pt() + j1.pt());

      // Find the appropriate mean pT bin and escape if needed
      const size_t njetBin = findPtBin(ptAvg/GeV);
      if (njetBin >= N_PT_BINS_dj) vetoEvent;

      // Now run the substructure algs...
      fastjet::PseudoJet filtered0 = _filter(j0);
      fastjet::PseudoJet filtered1 = _filter(j1);
      fastjet::PseudoJet trimmed0 = _trimmer(j0);
      fastjet::PseudoJet trimmed1 = _trimmer(j1);
      fastjet::PseudoJet pruned0 = _pruner(j0);
      fastjet::PseudoJet pruned1 = _pruner(j1);

      // ... and fill the histograms
      _h_ungroomedAvgJetMass_dj[njetBin]->fill(0.5*(j0.m() + j1.m())/GeV, weight);
      _h_filteredAvgJetMass_dj[njetBin]->fill(0.5*(filtered0.m() + filtered1.m())/GeV, weight);
      _h_trimmedAvgJetMass_dj[njetBin]->fill(0.5*(trimmed0.m() + trimmed1.m())/GeV, weight);
      _h_prunedAvgJetMass_dj[njetBin]->fill(0.5*(pruned0.m() + pruned1.m())/GeV, weight);
    }


    /// Normalise histograms etc., after the run
    void finalize() {
      const double normalizationVal = 1000;
      for (size_t i = 0; i < N_PT_BINS_dj; ++i) {
        normalize(_h_ungroomedAvgJetMass_dj[i], normalizationVal);
        normalize(_h_filteredAvgJetMass_dj[i], normalizationVal);
        normalize(_h_trimmedAvgJetMass_dj[i], normalizationVal);
        normalize(_h_prunedAvgJetMass_dj[i], normalizationVal);
      }
    }

    //@}


  private:

    /// @name FastJet grooming tools (configured in constructor init list)
    //@{
    const fastjet::Filter _filter;
    const fastjet::Filter _trimmer;
    const fastjet::Pruner _pruner;
    //@}


    /// @name Histograms
    //@{
    enum BINS_dj { PT_220_300_dj=0, PT_300_450_dj, PT_450_500_dj, PT_500_600_dj,
                   PT_600_800_dj, PT_800_1000_dj, PT_1000_1500_dj, N_PT_BINS_dj };
    Histo1DPtr _h_ungroomedJet0pt, _h_ungroomedJet1pt;
    Histo1DPtr _h_ungroomedAvgJetMass_dj[N_PT_BINS_dj];
    Histo1DPtr _h_filteredAvgJetMass_dj[N_PT_BINS_dj];
    Histo1DPtr _h_trimmedAvgJetMass_dj[N_PT_BINS_dj];
    Histo1DPtr _h_prunedAvgJetMass_dj[N_PT_BINS_dj];
    //@}


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(CMS_2013_I1224539_DIJET);


}
