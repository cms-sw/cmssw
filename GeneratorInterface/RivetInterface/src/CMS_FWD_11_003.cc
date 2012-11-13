// Samantha Dooling DESY
// February 2012
//
// -*- C++ -*-
// =============================
//
// Ratio of the energy deposited in the pseudorapditiy range 
// -6.6 < eta < -5.2 for events with a charged particle jet
//
// =============================
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/Beam.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "LWH/Histogram1D.h"

namespace Rivet {


  class CMS_FWD_11_003 : public Analysis {
  public:

  /// Constructor
  CMS_FWD_11_003()
    : Analysis("CMS_FWD_11_003")
    {
      setBeams(PROTON,PROTON);
    }

    // counters
    double evcounter_incl;
    double evcounter_jet;

    double norm_incl_eflow_09;
    double norm_incl_eflow_276;
    double norm_incl_eflow_7;

    void init() {

      //gives the range of eta and min pT for the final state from which I get the jets
      const ChargedFinalState fsj(-2.5,2.5,0.3*GeV);    
      addProjection(fsj, "FSJ"); 

      FastJets jetpro (fsj, FastJets::ANTIKT, 0.5);	
      jetpro.useInvisibles();
      addProjection(jetpro, "Jets");
    
      //gives the range of eta and min pT for the final state
      const FinalState fs(-7.0,-4.0,0.0*GeV);    
      addProjection(fs, "FS"); 
      VetoedFinalState fsv(fs);
      // skip Neutrinos and Muons
      fsv.vetoNeutrinos();	
      fsv.addVetoPairDetail(MUON, 0.0*GeV, 99999.9*GeV);
      addProjection(fsv, "fsv");
        
      // for the hadron level selection
      const FinalState sfs(-1000.0,1000.0,0.0*GeV);    
      addProjection(sfs, "sfs"); 
      VetoedFinalState sfsv(sfs);
      sfsv.vetoNeutrinos();
      sfsv.addVetoPairDetail(MUON, 0.0*GeV, 99999.9*GeV);
      addProjection(sfsv, "sfsv");
    
      //counters
      evcounter_incl  = 0;
      evcounter_jet = 0;

      // pt binning    
      // int  NptBins = 7;
      // double ptbinning[8] = {1.,2.,3.,5.,7.5,10,15,25};
	
      if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){

        // temporary histograms to fill the energy flow for inclusive events and leading jet events
        // in finalyze() I determine the ratios

        _tmp_incl_09.reset(new LWH::Histogram1D(binEdges(1,1,1))); // inclusive energy flow in the eta range of CASTOR
	_tmp_jet_09  = bookHistogram1D("eflow_jet_09",binEdges(2,1,1)); // Leading jet energy flow in pt
	_tmp_njet_09 = bookHistogram1D("number_jet_09",binEdges(2,1,1)); // Number of events in pt
	
      }

      if(fuzzyEquals(sqrtS()/GeV, 2760, 1E-3)){

        _tmp_incl_276.reset(new LWH::Histogram1D(binEdges(3,1,1)));
	_tmp_jet_276  = bookHistogram1D("eflow_jet_276",binEdges(4,1,1));
	_tmp_njet_276 = bookHistogram1D("number_jet_276",binEdges(4,1,1));
	
      }
  

      if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){ 

        _tmp_incl_7.reset(new LWH::Histogram1D(binEdges(5,1,1)));
        _tmp_jet_7  = bookHistogram1D("eflow_jet_7",binEdges(6,1,1));
        _tmp_njet_7 = bookHistogram1D("number_jet_7",binEdges(6,1,1));
	
     }

    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      const double weight = event.weight();

      // Skip if the event is empty
      const FinalState& fsv = applyProjection<FinalState>(event, "fsv");
      if (fsv.empty()) vetoEvent;
      
      const FastJets& jetpro = applyProjection<FastJets>(event, "Jets"); 
      const Jets& jets = jetpro.jetsByPt(1.0*GeV);
    
      // ====================== Minimum Bias selection
      //  ============================== xi cuts 
    
      bool passedHadronCuts = false;
    
      double xix    = 10;
      double xiy    = 10;
      double xidd   = 10e10;
      double Rapiditymax = -1;
    
      // calculate xi of the event
      // sort Particles in rapidity, from rapidity_min to rapidity_max
	
      ParticleVector myTempParticles;
      ParticleVector myRapiditySortedParticles;
      
      // copy only final stable particles in tempvector
      const FinalState& sfsv = applyProjection<FinalState>(event, "sfsv");
      if (sfsv.empty()) vetoEvent;

      foreach (const Particle& p, sfsv.particles()) {
        myTempParticles.push_back(Particle(p));
      }

      while (myTempParticles.size() != 0){
        double min_y = 10000;
	int min_y_pos = -1;
	for (unsigned int ipart = 0; ipart < myTempParticles.size(); ++ipart){
	  if (myTempParticles[ipart].momentum().rapidity() < min_y){
	    min_y = myTempParticles[ipart].momentum().rapidity();
	    min_y_pos = ipart;
	  }
	}
        myRapiditySortedParticles.push_back(myTempParticles[min_y_pos]);
	myTempParticles.erase(myTempParticles.begin()+min_y_pos);
      }

  			
      // find deltaymax
      double deltaymax  = 0;
      int deltaymax_pos = -1;
      for (unsigned int ipart=0; ipart < myRapiditySortedParticles.size()-1; ++ipart) {
        double deltay = myRapiditySortedParticles[ipart+1].momentum().rapidity() - myRapiditySortedParticles[ipart].momentum().rapidity();
        if (deltay > deltaymax) {
	  deltaymax     = deltay;
	  deltaymax_pos = ipart;
        }
      }
      Rapiditymax = deltaymax;
    
      // calculate Mx2 and My2 
      FourMomentum Xfourmom;
      FourMomentum Yfourmom;
        		
      for (int ipart=0; ipart <= deltaymax_pos; ++ipart) {
        Xfourmom += myRapiditySortedParticles[ipart].momentum();
      }
      if(FourMomentum(Xfourmom).mass2() <0 )
	vetoEvent;
    
      long double Mx2 = FourMomentum(Xfourmom).mass()*FourMomentum(Xfourmom).mass();

      for (unsigned int ipart = deltaymax_pos+1; ipart < myRapiditySortedParticles.size(); ++ipart) {
        Yfourmom += myRapiditySortedParticles[ipart].momentum();
      }
      if(FourMomentum(Yfourmom).mass2() <0 )
	vetoEvent;

      long double My2 = FourMomentum(Yfourmom).mass()*FourMomentum(Yfourmom).mass() ;

      // calculate xix and xiy and xidd
      xix  = Mx2/(sqrtS()/GeV*sqrtS()/GeV);
      xiy  = My2/(sqrtS()/GeV*sqrtS()/GeV);
      xidd = (Mx2*My2)/(sqrtS()/GeV*sqrtS()/GeV*0.938*0.938);

      // combine the selection
      if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)) {		
    	if(xix > 0.1  || xiy > 0.4 || xidd > 0.5) passedHadronCuts = true;
      }
      if(fuzzyEquals(sqrtS()/GeV, 2760, 1E-3)) {		
    	if(xix > 0.07 || xiy > 0.2 || xidd > 0.5) passedHadronCuts = true;
      }
      if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)) {		
    	if(xix > 0.04 || xiy > 0.1 || xidd > 0.5) passedHadronCuts = true;
      }

      // skip the event if the hadron cut is not fullfilled
      if(passedHadronCuts == false){
	vetoEvent;
      }
     
      //  ============================== MINIMUM BIAS EVENTS 					
  
      // loop over particles to calculate the energy 
      evcounter_incl += weight;
    
      foreach (const Particle& p, fsv.particles()) {

        if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)) {

	  _tmp_incl_09 -> 
	    fill(p.momentum().pseudorapidity(), weight * p.momentum().E()/GeV );
       }

	if(fuzzyEquals(sqrtS()/GeV, 2760, 1E-3)) {

	  _tmp_incl_276 -> 
	    fill(p.momentum().pseudorapidity(), weight * p.momentum().E()/GeV );
	}

     	if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)) {

	  _tmp_incl_7 -> 
	    fill(p.momentum().pseudorapidity(), weight * p.momentum().E()/GeV );
	}  
      }
        
      //  ============================== JET EVENTS  
         
      if(jets.size()>0){
      
        signed int index_1 = -1;	// for the jet with the 1.highest pT
        double tempmax_1   = -100;
         
        // jet with the 1.highest pt	
        for(signed int ijets = 0; ijets < (int)jets.size(); ++ijets){
    	  if(tempmax_1 == -100 || tempmax_1 < jets[ijets].momentum().pT()/GeV){	
    	    tempmax_1 = jets[ijets].momentum().pT()/GeV;
    	    index_1   = ijets;
    	  }		
     	}
 	
    	if(index_1 != -1 ){
 	
          // *******
     	  // 900 GeV 
     	  // ******* 
	
          if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){	
	  
            //eta cut for the central jets
      	    if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.0 ){

              // fill number of events
	      _tmp_njet_09-> 
       	        fill(jets[index_1].momentum().pT()/GeV, weight   );

	      // energy flow 						
              foreach (const Particle& p, fsv.particles()){	

	        // ask for the CASTOR region
      	   	if(-5.2 > p.momentum().pseudorapidity() && p.momentum().pseudorapidity() > -6.6 ){ 
		  _tmp_jet_09 -> 
	            fill(jets[index_1].momentum().pT()/GeV, weight * p.momentum().E()/GeV );
    	   	}
              } 
	    }// eta 
	  }// energy 

          // *******
          // 2760 GeV 
          // ******* 

	  if(fuzzyEquals(sqrtS()/GeV, 2760, 1E-3)){	

            // eta cut for the central jets
            if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.0 ){

	      // fill number of events
	      _tmp_njet_276-> 
                fill(jets[index_1].momentum().pT()/GeV, weight   );

              // energy flow 						
              foreach (const Particle& p, fsv.particles()){

	        // ask for the CASTOR region
      	   	if(-5.2 > p.momentum().pseudorapidity() && p.momentum().pseudorapidity() > -6.6 ){ 
		  _tmp_jet_276 -> 
	            fill(jets[index_1].momentum().pT()/GeV, weight * p.momentum().E()/GeV );
    	   	} 
              } 
	    }// eta 
	  }// energy

	  // *******
          // 7 TeV 
          // ******* 

	  if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){	

            // eta cut for the central jets
            if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.0 ){	
	       
              // fill number of events
	      _tmp_njet_7-> 
       	        fill(jets[index_1].momentum().pT()/GeV, weight   );

              // energy flow 						
              foreach (const Particle& p, fsv.particles()){	

	        // ask for the CASTOR region
     	   	if(-5.2 > p.momentum().pseudorapidity() && p.momentum().pseudorapidity() > -6.6 ){  
		  _tmp_jet_7 -> 
	            fill(jets[index_1].momentum().pT()/GeV, weight * p.momentum().E()/GeV );
    	   	}
              } 
             }// eta 
	    }// energy
          }// if jet index
        }// if jets

      }// analysis

    void finalize() {  

      AIDA::IHistogramFactory& hf = histogramFactory();

      const double norm_incl  = evcounter_incl ;
    
      if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){

        norm_incl_eflow_09 = _tmp_incl_09->binHeight(0)/norm_incl; // no normalization to the binwidth
	_tmp_jet_09->scale(1.0/norm_incl_eflow_09);

	//the energy flow ratio
        hf.divide(histoDir() + "/d02-x01-y01", *_tmp_jet_09, *_tmp_njet_09); 
	hf.destroy(_tmp_jet_09);
	hf.destroy(_tmp_njet_09);
       
       }
    
      if(fuzzyEquals(sqrtS()/GeV, 2760, 1E-3)){

	norm_incl_eflow_276 = _tmp_incl_276->binHeight(0)/norm_incl;
	_tmp_jet_276->scale(1.0/norm_incl_eflow_276);

	//the energy flow ratio
        hf.divide(histoDir() + "/d04-x01-y01", *_tmp_jet_276, *_tmp_njet_276); 
	hf.destroy(_tmp_jet_276);
	hf.destroy(_tmp_njet_276);
 
      }
   
      if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){

	norm_incl_eflow_7 = _tmp_incl_7->binHeight(0)/norm_incl;
	_tmp_jet_7->scale(1.0/norm_incl_eflow_7);

	//the energy flow ratio
        hf.divide(histoDir() + "/d06-x01-y01", *_tmp_jet_7, *_tmp_njet_7); 
	hf.destroy(_tmp_jet_7);
	hf.destroy(_tmp_njet_7);

      }

      MSG_INFO(" " );
      MSG_INFO("Number of inclusive events :  " << norm_incl );
    }

  private:
    shared_ptr<LWH::IHistogram1D> _tmp_incl_09;
    AIDA::IHistogram1D  *_tmp_jet_09,*_tmp_njet_09;    
 
    shared_ptr<LWH::IHistogram1D> _tmp_incl_276;
    AIDA::IHistogram1D  *_tmp_jet_276, *_tmp_njet_276;  

    shared_ptr<LWH::IHistogram1D> _tmp_incl_7;
    AIDA::IHistogram1D  *_tmp_jet_7, *_tmp_njet_7;
 
  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_FWD_11_003> plugin_CMS_FWD_11_003;


}
