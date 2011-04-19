// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/Beam.hh"

namespace Rivet {


  class CMS_2010_S8656010 : public Analysis {
  public:

    CMS_2010_S8656010() : Analysis("CMS_2010_S8656010") {
       setBeams(PROTON, PROTON);
       setNeedsCrossSection(false);
    }

//AK =====================================================INIT
    void init() {
      ChargedFinalState cfs(-2.5, 2.5, 0.1*GeV);
      addProjection(cfs, "CFS");
      addProjection(Beam(), "Beam");
 
      FinalState hfs(-5.5, 5.5, 0.1*GeV);
      addProjection(hfs, "HFS");

      _Nevt_after_cuts = 0;
	
      //eta bins
      _netabins=12;
      for (int ietabin=0; ietabin <= _netabins; ietabin++){	
	_etabins[ietabin]=0.2*ietabin;
      }
      
      
	//AK 7 TeV - pt spectra in eta bins
	_h_dNch_dpT_7[0] = bookHistogram1D(1, 1, 1);
	_h_dNch_dpT_7[1] = bookHistogram1D(1, 1, 2);
	_h_dNch_dpT_7[2] = bookHistogram1D(1, 1, 3);
	_h_dNch_dpT_7[3] = bookHistogram1D(1, 1, 4);
	
	_h_dNch_dpT_7[4] = bookHistogram1D(2, 1, 1);
	_h_dNch_dpT_7[5] = bookHistogram1D(2, 1, 2);
	_h_dNch_dpT_7[6] = bookHistogram1D(2, 1, 3);
	_h_dNch_dpT_7[7] = bookHistogram1D(2, 1, 4);
	
	_h_dNch_dpT_7[8] = bookHistogram1D(3, 1, 1);
	_h_dNch_dpT_7[9] = bookHistogram1D(3, 1, 2);
	_h_dNch_dpT_7[10] = bookHistogram1D(3, 1, 3);
	_h_dNch_dpT_7[11] = bookHistogram1D(3, 1, 4);
	
	_h_dNch_dEta_7 = bookHistogram1D(5, 1, 1);
	
    }

//AK =====================================================ANALYZE
    void analyze(const Event& event) {
      const double weight = event.weight();

      //charge particles
      const ChargedFinalState& charged = applyProjection<ChargedFinalState>(event, "CFS");
      if (charged.particles().size()<1) {
        vetoEvent;
         
      } 
      
      
      _Nevt_after_cuts += weight;
 
      foreach (const Particle& p, charged.particles()) {
        double pT = p.momentum().pT();      	
        double eta = p.momentum().eta();
	
	 if(fuzzyEquals(sqrtS(), 7000, 1E-3)){
           _h_dNch_dEta_7->fill(eta, weight);	       	
	   for (int ietabin=0; ietabin <= (_netabins-1); ietabin++){	
	      if (fabs(eta) < _etabins[ietabin+1] && fabs(eta) > _etabins[ietabin]){
              _h_dNch_dpT_7[ietabin]->fill(pT, weight);	  
	      }
            }
	 }
	 
      }
    }
    
//AK =====================================================FINALIZE
    void finalize() {
    	double normfac=1.0/_Nevt_after_cuts;  
	getLog() << Log::INFO << "Number of events after event selection: " << _Nevt_after_cuts << endl;	

 	if(fuzzyEquals(sqrtS(), 7000, 1E-3)){
     	   for (int ietabin=0; ietabin < _netabins; ietabin++){
            scale(_h_dNch_dpT_7[ietabin], normfac/(0.2*2.0)); //AK normalize to events and rapidity-bin
 	   }
           scale(_h_dNch_dEta_7, normfac);
	}
	
    }


//AK =====================================================DECLARATIONS
  private:


    AIDA::IHistogram1D* _h_dNch_dpT_7[12];
    AIDA::IHistogram1D* _h_dNch_dEta_7;
        
    int _netabins;
    double _Nevt_after_cuts;
    double _etabins[13];


   };


  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2010_S8656010> plugin_CMS_2010_S8656010;

}

