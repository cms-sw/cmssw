// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Tools/BinnedHistogram.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/RivetAIDA.hh"

namespace Rivet {

  // CMS azimuthal decorrelations
  class CMS_2011_S8950903 : public Analysis {
  public:

    CMS_2011_S8950903() : Analysis("CMS_2011_S8950903") {}


    void init() {
      FinalState fs;
      FastJets akt(fs, FastJets::ANTIKT, 0.5);
      addProjection(akt, "antikT");

      _h_deltaPhi.addHistogram(80., 110., bookHistogram1D(1, 1, 1));
      _h_deltaPhi.addHistogram(110., 140., bookHistogram1D(2, 1, 1));
      _h_deltaPhi.addHistogram(140., 200., bookHistogram1D(3, 1, 1));
      _h_deltaPhi.addHistogram(200., 300., bookHistogram1D(4, 1, 1));
      _h_deltaPhi.addHistogram(300., 7000., bookHistogram1D(5, 1, 1));
    }


    void analyze(const Event & event) {
      const double weight = event.weight();

      const Jets& jets = applyProjection<JetAlg>(event, "antikT").jetsByPt();
      if (jets.size() < 2) vetoEvent;

      if (fabs(jets[0].momentum().eta()) > 1.1 || jets[0].momentum().pT() < 80.) vetoEvent;
      if (fabs(jets[1].momentum().eta()) > 1.1 || jets[1].momentum().pT() < 30.) vetoEvent;

      double dphi = deltaPhi(jets[0].momentum(), jets[1].momentum().phi());

      _h_deltaPhi.fill(jets[0].momentum().pT(), dphi, weight);
    }


    void finalize() {
      foreach (AIDA::IHistogram1D* histo, _h_deltaPhi.getHistograms()) {
        normalize(histo, 1.);
      }
    }

  private:

    BinnedHistogram<double> _h_deltaPhi;

  };

  // This global object acts as a hook for the plugin system
//AK  DECLARE_RIVET_PLUGIN(CMS_2011_S8950903);
  AnalysisBuilder<CMS_2011_S8950903> plugin_CMS_2011_S8950903;

}

