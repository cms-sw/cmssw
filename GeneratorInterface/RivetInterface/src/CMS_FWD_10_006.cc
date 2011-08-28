// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Particle.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"

namespace Rivet {

  class CMS_FWD_10_006 : public Analysis {

  private:

    // Data members like post-cuts event weight counters go here

    AIDA::IHistogram1D* _hist_jetpt_forward;
    AIDA::IHistogram1D* _hist_jetpt_central;


  public:
    
    /// @name Constructors etc.
    //@{
    
    /// Constructor
    CMS_FWD_10_006()
      : Analysis("CMS_FWD_10_006")    {  
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }
    

    /// Book histograms and initialise projections before the run
    void init() {

      const FinalState fs(-25, 25, 0*GeV);      
      
      //Projections
      
      addProjection(fs, "FS");
      FastJets fj(fs, FastJets::ANTIKT, 0.5);
      addProjection(fj, "Jets");
      
      
      //Histograms
      
      _hist_jetpt_forward = bookHistogram1D(1,1,1);
      _hist_jetpt_central = bookHistogram1D(2,1,1);


    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      const double weight = event.weight();

      const FastJets& fastjets = applyProjection<FastJets>(event, "Jets"); 
      const Jets jets = fastjets.jetsByPt(32.);
      double cjet_pt=0.0;
      double fjet_pt=0.0;

      foreach( const Jet& j, jets ) {
	double pT = 0.0;
	pT = j.momentum().pT();
	if( pT < 35 ) continue;
	
	if ( fabs(j.momentum().eta()) < 2.8 ) {
	  if(cjet_pt<pT)
	    cjet_pt = pT;
	}
	
	if ( ( fabs(j.momentum().eta()) < 4.7 ) && ( fabs(j.momentum().eta()) > 3.2 ) ) {
	  if(fjet_pt<pT)
	    fjet_pt = pT;
	}
	
      }

      if( cjet_pt>35 && fjet_pt>35 ){
	_hist_jetpt_forward->fill( fjet_pt, weight );
	_hist_jetpt_central->fill( cjet_pt, weight );
      }


    }


    /// Normalise histograms etc., after the run
    void finalize() {

      // value of picobarn is one.
      double invlumi = crossSection()/picobarn/sumOfWeights();

      // Scale with the width of the used eta bin, weights and cross section.
      scale(_hist_jetpt_forward, invlumi/3.0); 
      scale(_hist_jetpt_central, invlumi/5.6);
     
    }
    
    //@}



  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_FWD_10_006> plugin_CMS_FWD_10_006;


}
