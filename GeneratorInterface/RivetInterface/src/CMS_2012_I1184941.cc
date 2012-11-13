// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Tools/Logging.hh"

namespace Rivet {


  class CMS_2012_I1184941 : public Analysis {
  public:

    CMS_2012_I1184941()
      : Analysis("CMS_2012_I1184941")
    {   }

  public:

    void init() {

      FinalState fs;			
      addProjection(fs, "FS");

      const FastJets jets(FinalState(-4.4, 4.4, 0.0*GeV), FastJets::ANTIKT, 0.5);
      addProjection(jets, "AntiKtJets05");

      _hist_1 = bookHistogram1D(1, 1, 1);

    }

    void analyze(const Event& event) {

      const double weight = event.weight();

      double xiM = 0.;
      double xiP = 0.;
      const double jetptCut = 20*GeV;

      const Jets jets = applyProjection < FastJets > (event, "AntiKtJets05").jetsByPt();

      MSG_DEBUG("jet size = " << jets.size());

      if (jets.size() < 2) vetoEvent;  //require a dijet system

      MSG_DEBUG("jet1 pt = " << jets[1].momentum().pT());

      if (jets[1].momentum().pT() < jetptCut) vetoEvent; // cut on leading and 2nd leading jet. 

      const FinalState& fsp = applyProjection < FinalState > (event, "FS");

      foreach(const Particle& p, fsp.particlesByEta())
      {
                double eta = p.momentum().eta();
                double energy = p.momentum().E();
                double costheta = cos(p.momentum().theta());

		if ( eta < 4.9 ) {
			xiP += (energy + energy*costheta);
		}
		if ( -4.9 < eta ) {
			xiM += (energy - energy*costheta);
		}
      }
	
      xiP = xiP / (sqrtS()/GeV);
      xiM = xiM / (sqrtS()/GeV);

      _hist_1->fill( xiM, weight ); // Fill the histogram both with xiP and xiM, and get the average in the endjob.
      _hist_1->fill( xiP, weight );

    }

    void finalize() {

      scale( _hist_1, crossSection()/microbarn/sumOfWeights() / 2.);

    }

  private:

    AIDA::IHistogram1D* _hist_1;

  };

  // The hook for the plugin system
  //AK DECLARE_RIVET_PLUGIN(CMS_2012_I1184941);
  AnalysisBuilder<CMS_2012_I1184941> plugin_CMS_2012_I1184941;

}
