// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/Beam.hh"

namespace Rivet {


  class CMS_2010_S8547297 : public Analysis {
  public:

    CMS_2010_S8547297() : Analysis("CMS_2010_S8547297") {
       setBeams(PROTON, PROTON);
       setNeedsCrossSection(false);
    }

//AK =====================================================INIT
    void init() {
      ChargedFinalState cfs(-2.5, 2.5, 0.0*GeV);
      addProjection(cfs, "CFS");
      addProjection(Beam(), "Beam");

      _Nevt_after_cuts = 0;
      
      if(fuzzyEquals(sqrtS(), 900*GeV, 1E-3)){
	_h_dNch_dpT[0] = bookHistogram1D(1, 1, 1);
	_h_dNch_dpT[1] = bookHistogram1D(1, 1, 2);
	_h_dNch_dpT[2] = bookHistogram1D(1, 1, 3);
	_h_dNch_dpT[3] = bookHistogram1D(1, 1, 4);

	_h_dNch_dpT[4] = bookHistogram1D(2, 1, 1);
	_h_dNch_dpT[5] = bookHistogram1D(2, 1, 2);
	_h_dNch_dpT[6] = bookHistogram1D(2, 1, 3);
	_h_dNch_dpT[7] = bookHistogram1D(2, 1, 4);

	_h_dNch_dpT[8] = bookHistogram1D(3, 1, 1);
	_h_dNch_dpT[9] = bookHistogram1D(3, 1, 2);
	_h_dNch_dpT[10] = bookHistogram1D(3, 1, 3);
	_h_dNch_dpT[11] = bookHistogram1D(3, 1, 4);

	_h_dNch_dpT_all = bookHistogram1D(7, 1, 1);

	_h_dNch_dEta = bookHistogram1D(8, 1, 1);
      } else if (fuzzyEquals(sqrtS(), 2360*GeV, 1E-3)){
	_h_dNch_dpT[0] = bookHistogram1D(4, 1, 1);
	_h_dNch_dpT[1] = bookHistogram1D(4, 1, 2);
	_h_dNch_dpT[2] = bookHistogram1D(4, 1, 3);
	_h_dNch_dpT[3] = bookHistogram1D(4, 1, 4);

	_h_dNch_dpT[4] = bookHistogram1D(5, 1, 1);
	_h_dNch_dpT[5] = bookHistogram1D(5, 1, 2);
	_h_dNch_dpT[6] = bookHistogram1D(5, 1, 3);
	_h_dNch_dpT[7] = bookHistogram1D(5, 1, 4);

	_h_dNch_dpT[8] = bookHistogram1D(6, 1, 1);
	_h_dNch_dpT[9] = bookHistogram1D(6, 1, 2);
	_h_dNch_dpT[10] = bookHistogram1D(6, 1, 3);
	_h_dNch_dpT[11] = bookHistogram1D(6, 1, 4);

	_h_dNch_dpT_all = bookHistogram1D(7, 1, 2);

	_h_dNch_dEta = bookHistogram1D(8, 1, 2);
      }
      return;
    }

//AK =====================================================ANALYZE
    void analyze(const Event& event) {

      if (!fuzzyEquals(sqrtS(), 900*GeV, 1E-3) && !fuzzyEquals(sqrtS(), 2360*GeV, 1E-3) ){
	return;
      }

      const double weight = event.weight();

      //charge particles
      const ChargedFinalState& charged = applyProjection<ChargedFinalState>(event, "CFS");
      
      _Nevt_after_cuts += weight;
 
      foreach (const Particle& p, charged.particles()) {
        const double pT = p.momentum().pT();      	
        const double eta = p.momentum().eta();
	
	// The data is actually a duplicated folded distribution.  This should mimic it.
	_h_dNch_dEta->fill(eta, 0.5*weight);
	_h_dNch_dEta->fill(-eta, 0.5*weight);
	if (fabs(eta)<2.4 && pT>0.1) {
	  if (pT<4.0) {
	    _h_dNch_dpT_all->fill(pT, weight/pT);
	    if (pT<2.0) {
	      int ietabin = fabs(eta)/0.2;
	      _h_dNch_dpT[ietabin]->fill(pT, weight);
	    }
	  }
	}
      }
      return;
    }
    
//AK =====================================================FINALIZE
    void finalize() {

      if (!fuzzyEquals(sqrtS(), 900*GeV, 1E-3) && !fuzzyEquals(sqrtS(), 2360*GeV, 1E-3) ){
	return;
      }

      const double normfac = 1.0/_Nevt_after_cuts; // Normalizing to unit eta is automatic
      // The pT distributions in bins of eta must be normalized to unit eta.  This is a factor of 2
      // for the |eta| times 0.2 (eta range).
      // The pT distributions over all eta are normalized to unit eta (2.0*2.4) and by 1/2*pi*pT. 
      // The 1/pT part is taken care of in the filling.  The 1/2pi is taken care of here.
      const double normpT = normfac/(2.0*0.2);
      const double normpTall = normfac/(2.0*3.141592654*2.0*2.4);

      for (int ietabin=0; ietabin < 12; ietabin++){
	scale(_h_dNch_dpT[ietabin], normpT);
      }
      scale(_h_dNch_dpT_all, normpTall);
      scale(_h_dNch_dEta, normfac);
      return;
    }


//AK =====================================================DECLARATIONS
  private:

    AIDA::IHistogram1D* _h_dNch_dpT[12];
    AIDA::IHistogram1D* _h_dNch_dpT_all;

    AIDA::IHistogram1D* _h_dNch_dEta;
        
    double _Nevt_after_cuts;


   };


  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2010_S8547297> plugin_CMS_2010_S8547297;

}

