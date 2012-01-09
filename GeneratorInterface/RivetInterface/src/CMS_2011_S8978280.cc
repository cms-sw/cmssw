// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/UnstableFinalState.hh"
#include "Rivet/Projections/Beam.hh"

namespace Rivet {

  /// @brief CMS strange particle spectra (Ks, Lambda, Cascade) in pp at 900 and 7000 GeV
  /// @author Kevin Stenson
  class CMS_2011_S8978280 : public Analysis {
  public:

    /// Constructor
    CMS_2011_S8978280()
      : Analysis("CMS_2011_S8978280"),
	_Nevt_after_cuts(0.0)
    {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(false);
    }


    void init() {

      addProjection(Beam(), "Beam");

      // Need wide range of eta because cut on rapidity not pseudorapidity
      UnstableFinalState ufs(-8.0, 8.0, 0.0*GeV);
      addProjection(ufs, "UFS");

      // Particle distributions versus rapidity and transverse momentum
      // Only make histograms if the correct energy is used.
      if (fuzzyEquals(sqrtS(), 900*GeV, 1E-3)){
	_h_dNKshort_dy = bookHistogram1D(1, 1, 1);
	_h_dNKshort_dpT = bookHistogram1D(2, 1, 1);
	_h_dNLambda_dy = bookHistogram1D(3, 1, 1);
	_h_dNLambda_dpT = bookHistogram1D(4, 1, 1);
	_h_dNXi_dy = bookHistogram1D(5, 1, 1);
	_h_dNXi_dpT = bookHistogram1D(6, 1, 1);
      } else if (fuzzyEquals(sqrtS(), 7000*GeV, 1E-3)){
	_h_dNKshort_dy = bookHistogram1D(1, 1, 2);
	_h_dNKshort_dpT = bookHistogram1D(2, 1, 2);
	_h_dNLambda_dy = bookHistogram1D(3, 1, 2);
	_h_dNLambda_dpT = bookHistogram1D(4, 1, 2);
	_h_dNXi_dy = bookHistogram1D(5, 1, 2);
	_h_dNXi_dpT = bookHistogram1D(6, 1, 2);
      }
      return;
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      if (!fuzzyEquals(sqrtS(), 900*GeV, 1E-3) && !fuzzyEquals(sqrtS(), 7000*GeV, 1E-3) ){
	return;
      }
      const double weight = event.weight();

      _Nevt_after_cuts += weight;

      // This works as long as the KShort, Lambda, and Cascade are not decayed in the generator.
      const UnstableFinalState& parts = applyProjection<UnstableFinalState>(event, "UFS");

      foreach (const Particle& p, parts.particles()) {
        const double pT = p.momentum().pT();
        const double y = fabs(p.momentum().rapidity());
	const PdgId pid = abs(p.pdgId());

	if (y < 2.0) {

	  switch (pid) {
	  case K0S:
	      _h_dNKshort_dy->fill(y, weight);
	      _h_dNKshort_dpT->fill(pT, weight);
	    break;
	  case LAMBDA:
	    // Lambda should not have Cascade or Omega ancestors since they should not decay. But just in case...
	    if ( !( p.hasAncestor(3322) || p.hasAncestor(-3322) || p.hasAncestor(3312) || p.hasAncestor(-3312) || p.hasAncestor(3334) || p.hasAncestor(-3334) ) ) {
	      _h_dNLambda_dy->fill(y, weight);
	      _h_dNLambda_dpT->fill(pT, weight);
		 }
	    break;
	  case XIMINUS:
	    // Cascade should not have Omega ancestors since it should not decay.  But just in case...
	    if ( !( p.hasAncestor(3334) || p.hasAncestor(-3334) ) ) {
	      _h_dNXi_dy->fill(y, weight);
	      _h_dNXi_dpT->fill(pT, weight);
	    }
	    break;
	  }
	}
      }
      return;
    }


    void finalize() {
      if (!fuzzyEquals(sqrtS(), 900*GeV, 1E-3) && !fuzzyEquals(sqrtS(), 7000*GeV, 1E-3) ){
	return;
      }
      AIDA::IHistogramFactory& hf = histogramFactory();
      const string dir = histoDir();

      // Making the Lambda/Kshort and Xi/Lambda ratios vs pT and y
      if (fuzzyEquals(sqrtS(), 900*GeV, 1E-3)){
	hf.divide(dir + "/d07-x01-y01",*_h_dNLambda_dpT, *_h_dNKshort_dpT);
	hf.divide(dir + "/d08-x01-y01",*_h_dNXi_dpT, *_h_dNLambda_dpT);
	hf.divide(dir + "/d09-x01-y01",*_h_dNLambda_dy, *_h_dNKshort_dy);
	hf.divide(dir + "/d10-x01-y01",*_h_dNXi_dy, *_h_dNLambda_dy);
      } else if (fuzzyEquals(sqrtS(), 7000*GeV, 1E-3)){
	hf.divide(dir + "/d07-x01-y02",*_h_dNLambda_dpT, *_h_dNKshort_dpT);
	hf.divide(dir + "/d08-x01-y02",*_h_dNXi_dpT, *_h_dNLambda_dpT);
	hf.divide(dir + "/d09-x01-y02",*_h_dNLambda_dy, *_h_dNKshort_dy);
	hf.divide(dir + "/d10-x01-y02",*_h_dNXi_dy, *_h_dNLambda_dy);
      }

      double normpT = 1.0/_Nevt_after_cuts;
      double normy = 0.5*normpT; // Accounts for using |y| instead of y
      scale(_h_dNKshort_dy, normy);
      scale(_h_dNKshort_dpT, normpT);
      scale(_h_dNLambda_dy, normy);
      scale(_h_dNLambda_dpT, normpT);
      scale(_h_dNXi_dy, normy);
      scale(_h_dNXi_dpT, normpT);

      return;
    }


  private:

  double _Nevt_after_cuts;

    // Particle distributions versus rapidity and transverse momentum
    AIDA::IHistogram1D *_h_dNKshort_dy;
    AIDA::IHistogram1D *_h_dNKshort_dpT;
    AIDA::IHistogram1D *_h_dNLambda_dy;
    AIDA::IHistogram1D *_h_dNLambda_dpT;
    AIDA::IHistogram1D *_h_dNXi_dy;
    AIDA::IHistogram1D *_h_dNXi_dpT;

  };


  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8978280> plugin_CMS_2011_S8978280;


}
