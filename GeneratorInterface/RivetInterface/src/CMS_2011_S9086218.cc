// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Tools/BinnedHistogram.hh"

namespace Rivet {

  // This analysis is a derived from the class Analysis:
  class CMS_2011_S9086218 : public Analysis {

 
  private:

    // Initialize a histogram:
    BinnedHistogram<double> _hist_sigma;

  public:
    // @name Constructors, init, analyze, finalize
    // @{

    // Constructor
    CMS_2011_S9086218()
    // This name must be the same as the one you cal in rivet_cfg.py
      : Analysis("CMS_2011_S9086218") {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }

    // Book histograms and initialize projections:
    void init() {
      
      const FinalState fs;

      // Initialize the projectors:
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.5),"Jets");

      // Book histograms:
      _hist_sigma.addHistogram(0.0, 0.5, bookHistogram1D(1, 1, 1));
      _hist_sigma.addHistogram(0.5, 1.0, bookHistogram1D(2, 1, 1));
      _hist_sigma.addHistogram(1.0, 1.5, bookHistogram1D(3, 1, 1));
      _hist_sigma.addHistogram(1.5, 2.0, bookHistogram1D(4, 1, 1));
      _hist_sigma.addHistogram(2.0, 2.5, bookHistogram1D(5, 1, 1));
      _hist_sigma.addHistogram(2.5, 3.0, bookHistogram1D(6, 1, 1));


    }

    // Analysis
    void analyze(const Event &event) {

      // for most generators weight = 1, but in some cases it might be different.
      const double weight = event.weight();
      
      // Apply the projection:
      const FastJets &fj = applyProjection<FastJets>(event,"Jets");
      
      // Get the jets out with 18 GeV < p_T < 1100 GeV and |y| < 4.7.
      // Note that FastJets.jets by default uses ETA, but it can also be
      // changed to RAPIDITY.
      const Jets& jets = fj.jets(18*GeV, 1100*GeV, -4.7, 4.7, RAPIDITY);

      // Fill the relevant histograms:
      foreach(const Jet &j, jets) {
        _hist_sigma.fill(fabs(j.momentum().rapidity()), j.momentum().pT(), weight);
      }
    }

    // Finalize
    void finalize() {

      // Normalize to the width of etabin:
      _hist_sigma.scale(crossSection()/sumOfWeights()/0.5/2, this);
    }
    //@}


  };

  // This global object acts as a hook for the plugin system. 
  AnalysisBuilder<CMS_2011_S9086218> plugin_CMS_2011_S9086218;

}
