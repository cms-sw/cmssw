// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"


namespace Rivet {

  // This analysis is a derived from the class Analysis:
  class CMS_FWD_10_003 : public Analysis {

 
  private:

    // Initialize a histogram:
    AIDA::IHistogram1D *_hist_sigma;

  public:
    // @name Constructors, init, analyze, finalize
    // @{

    // Constructor
    CMS_FWD_10_003()
    // This name must be the same as the one you call in rivet_cfg.py
      : Analysis("CMS_FWD_10_003") {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }

    // Book histograms and initialize projections:
    void init() {
      
      const FinalState fs;

      // Initialize the projectors:
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.5),"Jets");

      // Book histograms:
      _hist_sigma = bookHistogram1D(1,1,2);
    }

    // Analysis
    void analyze(const Event &event) {

      // for most generators weight = 1, but in some cases it might be different.
      const double weight = event.weight();
      
      // Apply the projection:
      const FastJets &fj = applyProjection<FastJets>(event,"Jets");
      
      // Get the jets out with 35 GeV < p_T < 150 GeV and |eta| < 4.7.
      // Note that FastJets.jets by default uses ETA, but it can also be
      // changed to RAPIDITY.
      const Jets jets = fj.jets(35*GeV, 150*GeV, -4.7, 4.7, ETA);

      // Fill the histograms if 3.2 < |eta| < 4.7.
      foreach(const Jet &j, jets) {
        if(j.momentum().eta() > 3.2 || j.momentum().eta() < -3.2){
          _hist_sigma->fill(j.momentum().pT(),weight);
        }
      }

    }

    // Finalize
    void finalize() {
      const double deltaeta = 1.5;
      // crossSection() is by default in picobarn.
      scale(_hist_sigma, crossSection()/sumOfWeights()/deltaeta/2);
    }
    //@}


  };

  // This global object acts as a hook for the plugin system.
  AnalysisBuilder<CMS_FWD_10_003> plugin_CMS_FWD_10_003;

}

