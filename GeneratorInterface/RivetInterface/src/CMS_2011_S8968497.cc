// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Tools/BinnedHistogram.hh"

namespace Rivet {

  class CMS_2011_S8968497 : public Analysis {
  public:
    CMS_2011_S8968497() : Analysis("CMS_2011_S8968497") {
       setBeams(PROTON, PROTON);
    }
//=====================================================INIT
    void init() {
      FinalState fs;
      FastJets antikt(fs, FastJets::ANTIKT, 0.5);
      addProjection(antikt, "ANTIKT");
      _h_chi_dijet.addHistogram(2200., 7000., bookHistogram1D(1, 1, 1));
      _h_chi_dijet.addHistogram(1800., 2200., bookHistogram1D(2, 1, 1));
      _h_chi_dijet.addHistogram(1400., 1800., bookHistogram1D(3, 1, 1));
      _h_chi_dijet.addHistogram(1100., 1400., bookHistogram1D(4, 1, 1));
      _h_chi_dijet.addHistogram(850., 1100., bookHistogram1D(5, 1, 1));
      _h_chi_dijet.addHistogram(650., 850., bookHistogram1D(6, 1, 1));
      _h_chi_dijet.addHistogram(500., 650., bookHistogram1D(7, 1, 1));
      _h_chi_dijet.addHistogram(350., 500., bookHistogram1D(8, 1, 1));
      _h_chi_dijet.addHistogram(250., 350., bookHistogram1D(9, 1, 1));
    }
//=====================================================ANALYZE
    void analyze(const Event& event) {
      const double weight = event.weight();
      const Jets& jets = applyProjection<JetAlg>(event, "ANTIKT").jetsByPt();
      if (jets.size() < 2) vetoEvent;
      FourMomentum j0(jets[0].momentum());
      FourMomentum j1(jets[1].momentum());
      double y0 = j0.rapidity();
      double y1 = j1.rapidity();
      if (fabs(y0+y1)/2. > 1.11) vetoEvent;
      double mjj = FourMomentum(j0+j1).mass();
      double chi = exp(fabs(y0-y1));
      _h_chi_dijet.fill(mjj, chi, weight);

    }
//=====================================================FINALIZE
    void finalize() {
      foreach (AIDA::IHistogram1D* hist, _h_chi_dijet.getHistograms()) {
        normalize(hist);
      }
    }
//=====================================================DECLARATIONS
  private:
    BinnedHistogram<double> _h_chi_dijet;
   };
  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8968497> plugin_CMS_2011_S8968497;
}

