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
      
      
	//AK 900 GeV - pt spectra in eta bins
	_h_dNch_dpT_900[0] = bookHistogram1D(1, 1, 1);
	_h_dNch_dpT_900[1] = bookHistogram1D(1, 1, 2);
	_h_dNch_dpT_900[2] = bookHistogram1D(1, 1, 3);
	_h_dNch_dpT_900[3] = bookHistogram1D(1, 1, 4);
	
	_h_dNch_dpT_900[4] = bookHistogram1D(2, 1, 1);
	_h_dNch_dpT_900[5] = bookHistogram1D(2, 1, 2);
	_h_dNch_dpT_900[6] = bookHistogram1D(2, 1, 3);
	_h_dNch_dpT_900[7] = bookHistogram1D(2, 1, 4);
	
	_h_dNch_dpT_900[8] = bookHistogram1D(3, 1, 1);
	_h_dNch_dpT_900[9] = bookHistogram1D(3, 1, 2);
	_h_dNch_dpT_900[10] = bookHistogram1D(3, 1, 3);
	_h_dNch_dpT_900[11] = bookHistogram1D(3, 1, 4);
	
	//AK 2360 GeV - pt spectra in eta bins
	_h_dNch_dpT_2360[0] = bookHistogram1D(4, 1, 1);
	_h_dNch_dpT_2360[1] = bookHistogram1D(4, 1, 2);
	_h_dNch_dpT_2360[2] = bookHistogram1D(4, 1, 3);
	_h_dNch_dpT_2360[3] = bookHistogram1D(4, 1, 4);
	
	_h_dNch_dpT_2360[4] = bookHistogram1D(5, 1, 1);
	_h_dNch_dpT_2360[5] = bookHistogram1D(5, 1, 2);
	_h_dNch_dpT_2360[6] = bookHistogram1D(5, 1, 3);
	_h_dNch_dpT_2360[7] = bookHistogram1D(5, 1, 4);
	
	_h_dNch_dpT_2360[8] = bookHistogram1D(6, 1, 1);
	_h_dNch_dpT_2360[9] = bookHistogram1D(6, 1, 2);
	_h_dNch_dpT_2360[10] = bookHistogram1D(6, 1, 3);
	_h_dNch_dpT_2360[11] = bookHistogram1D(6, 1, 4);
	
	
	//Eta spectra
	_h_dNch_dEta_900 = bookHistogram1D(8, 1, 1);
	_h_dNch_dEta_2360 = bookHistogram1D(8, 1, 2);
	
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
	
	 if(fuzzyEquals(sqrtS(), 900, 1E-3)){
           _h_dNch_dEta_900->fill(eta, weight);	       	
	   for (int ietabin=0; ietabin <= (_netabins-1); ietabin++){	
	      if (fabs(eta) < _etabins[ietabin+1] && fabs(eta) > _etabins[ietabin]){
              _h_dNch_dpT_900[ietabin]->fill(pT, weight);	  
	      }
            }
	 }
	 
	 if(fuzzyEquals(sqrtS(), 2360, 1E-3)){
            _h_dNch_dEta_2360->fill(eta, weight);	       	
	    for (int ietabin=0; ietabin <= (_netabins-1); ietabin++){	
	       if (fabs(eta) < _etabins[ietabin+1] && fabs(eta) > _etabins[ietabin]){
               _h_dNch_dpT_2360[ietabin]->fill(pT, weight);	  
	       }
            }
	 }

      }
    }
    
//AK =====================================================FINALIZE
    void finalize() {
    	double normfac=1.0/_Nevt_after_cuts;  

	getLog() << Log::INFO << "Number of events after event selection: " << _Nevt_after_cuts << endl;	

 	if(fuzzyEquals(sqrtS(), 900, 1E-3)){
     	   for (int ietabin=0; ietabin < _netabins; ietabin++){
            scale(_h_dNch_dpT_900[ietabin], normfac/(0.2*2.0)); //AK normalize to events and rapidity-bin
 	   }
           scale(_h_dNch_dEta_900, normfac);
	}

	if(fuzzyEquals(sqrtS(), 2360, 1E-3)){
      	   for (int ietabin=0; ietabin < _netabins; ietabin++){
              scale(_h_dNch_dpT_2360[ietabin], normfac/(0.2*2.0)); //AK normalize to events and rapidity-bin
	   }
           scale(_h_dNch_dEta_2360, normfac);
	}
	
    }


//AK =====================================================DECLARATIONS
  private:


    AIDA::IHistogram1D* _h_dNch_dpT_900[12];
    AIDA::IHistogram1D* _h_dNch_dpT_2360[12];

    AIDA::IHistogram1D* _h_dNch_dEta_900;
    AIDA::IHistogram1D* _h_dNch_dEta_2360;
        
    int _netabins;
    double _Nevt_after_cuts;
    double _etabins[13];


   };


  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2010_S8547297> plugin_CMS_2010_S8547297;

}

