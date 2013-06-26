// -*- C++ -*-

#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/Beam.hh"
#include "Rivet/Projections/VetoedFinalState.hh"

namespace Rivet 
{
  class CMS_2011_I930319 : public Analysis 
  {
  
  public:
    

    /// Constructor    
    CMS_2011_I930319()
      : Analysis("CMS_2011_I930319")
      
      {
      	setBeams(PROTON,PROTON);
      }
           	
	//counters 
	double evcounter_mb;
 	double evcounter_dijet;
 

  void init() {
    const FinalState fs(-6.0,6.0,0.0*GeV);             
    addProjection(fs, "FS"); 
    addProjection(FastJets(fs, FastJets::ANTIKT, 0.5), "Jets");	
    
    VetoedFinalState fsv(fs);
    fsv.vetoNeutrinos();
    fsv.addVetoPairDetail(MUON, 0.0*GeV, 99999.9*GeV);
    addProjection(fsv, "fsv");
        
    // for the MB ND selection
    const ChargedFinalState fschrgd(-6.0,6.0,0.0*GeV);             
    addProjection(fschrgd, "fschrgd"); 
    VetoedFinalState fschrgdv(fschrgd);
    fschrgdv.vetoNeutrinos();
    addProjection(fschrgdv, "fschrgdv");
    
    evcounter_mb = 0;
    evcounter_dijet = 0;
    
    if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){
      _hist_mb_09 	 = bookHistogram1D(1,1,1); // energy flow in MB, 0.9 TeV
      _hist_dijet_09 	 = bookHistogram1D(2,1,1); // energy flow in dijet events, 0.9 TeV
    }

    if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){     
      _hist_mb_7 	 = bookHistogram1D(3,1,1); // energy flow in MB, 7 TeV
      _hist_dijet_7 	 = bookHistogram1D(4,1,1); // energy flow in dijet events, 7 TeV
    }

  }

  void analyze(const Event& event) {
    const double weight = event.weight();     
    
    // Skip if the event is empty
    const FinalState& fsv = applyProjection<FinalState>(event, "fsv");
    if (fsv.empty()) vetoEvent;
      
    
    
    // Veto diffraction according to defined hadron level.
    double count_chrg_forward = 0;
    double count_chrg_backward = 0;
    const FinalState& fschrgdv = applyProjection<FinalState>(event, "fschrgdv");

    foreach (const Particle& p, fschrgdv.particles()) {
      if( 3.9 < p.momentum().pseudorapidity() && 
          p.momentum().pseudorapidity() < 4.4){count_chrg_forward++;}
      if(-4.4 < p.momentum().pseudorapidity() && 
         p.momentum().pseudorapidity() < -3.9){count_chrg_backward++;}
      
    }	

    if(count_chrg_forward*count_chrg_backward==0) {
      vetoEvent;
    }
    
    const FastJets& jetpro = applyProjection<FastJets>(event, "Jets"); 
    const Jets& jets = jetpro.jetsByPt(7.0*GeV);
    
	
    //  ============================== MINIMUM BIAS EVENTS 					
  
    //loop over particles to calculate the energy
    evcounter_mb += weight;
    foreach (const Particle& p, fsv.particles()) {

      if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)) {_hist_mb_09 -> 
          fill(fabs(p.momentum().pseudorapidity()), weight * p.momentum().E()/GeV );}

      if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)) {_hist_mb_7 -> 
          fill(fabs(p.momentum().pseudorapidity()), weight * p.momentum().E()/GeV );}  

    } 
    
    
    //  ============================== DIJET EVENTS  
         
    if(jets.size()>1){
      
      signed int index_1 = -1;	//for the jet with the 1.highest pT
      signed int index_2 = -1;	//for the jet with the 2.highest pT
      
      double tempmax_1 = -100;
      double tempmax_2 = -100;
      
      //jet with the 1.highest pt	
      for(signed int ijets = 0; ijets < (int)jets.size(); ijets++){
        if(tempmax_1 == -100 || tempmax_1 < jets[ijets].momentum().pT()/GeV)
          {	
            tempmax_1 = jets[ijets].momentum().pT()/GeV;
            index_1 = ijets;
          }		
      }
      
          
      //jet with the 2. highest pt	
      for(signed int ijets = 0; ijets < (int)jets.size(); ijets++){		
        if(tempmax_2 == -100 || tempmax_2 < jets[ijets].momentum().pT()/GeV){	
          if(jets[ijets].momentum().pT()/GeV < tempmax_1){
            tempmax_2 = jets[ijets].momentum().pT()/GeV;
            index_2 = ijets;
          }
        }				
      }
      
      
      if(index_1 != -1 && index_2 != -1){
 
        double diffphi = jets[index_2].momentum().phi() - jets[index_1].momentum().phi();    				 
        if(diffphi < -PI){ diffphi += 2.0*PI; }
        if(diffphi > PI){ diffphi -= 2.0*PI; }
        diffphi = fabs(diffphi);
        
	
        // *******
        // 900 GeV 
        // *******
        
	if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){	

          //pt cut
          if(jets[index_1].momentum().pT()/GeV > 8.0 && jets[index_2].momentum().pT()/GeV >8.0){				

            //eta cut for the central jets
            if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.5 && 
               fabs(jets[index_2].momentum().pseudorapidity()) < 2.5){
              
              //back to back condition of the jets
              if(fabs(diffphi-PI) < 1.0){	
                evcounter_dijet += weight;						
                
                //  E-flow 						
                foreach (const Particle& p, fsv.particles()){	
                  _hist_dijet_09->
                    fill(fabs(p.momentum().pseudorapidity()), weight*p.momentum().E()/GeV);
                
                }//foreach particle        
              }//if(dphi)					
            }// else (eta cut central region)
          }//pt cut			
        }// energy 
        
              
             	      
        // ********	
        // 7000 GeV		
        // ********
       
        if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){
          
          //pt cut
          if(jets[index_1].momentum().pT()/GeV > 20.0 && jets[index_2].momentum().pT()/GeV > 20.0){	

            //eta cut for the central jets
            if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.5 && 
               fabs(jets[index_2].momentum().pseudorapidity()) < 2.5){
              
              //back to back condition of the jets
              if(fabs(diffphi-PI) < 1.0){	
                evcounter_dijet += weight;
                
                //E-flow																								
                foreach (const Particle& p, fsv.particles()){								
                  _hist_dijet_7->
                    fill(fabs(p.momentum().pseudorapidity()), weight*p.momentum().E()/GeV);
    
                }//foreach particle
              }//if(dphi)
            }// else (eta cut central region)
          }//pt cut
        }// energy 			
	
      }// if index        
    }// analysis 
  }

    
  void finalize() {      
      
    const double norm_dijet = evcounter_dijet*2.0 ; //AK norm factor 2 for the +/- region
    const double norm_mb = evcounter_mb*2.0 ;
    
    if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){
        scale(_hist_mb_09, 1.0/norm_mb);
        scale(_hist_dijet_09, 1.0/norm_dijet);
    }

    if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){
      scale(_hist_dijet_7, 1.0/norm_dijet);
      scale(_hist_mb_7, 1.0/norm_mb);
    }	       
    
    getLog() << Log::INFO << " " << endl;
    getLog() << Log::INFO << "Number of MB events:  " << norm_mb << endl;
    getLog() << Log::INFO << "Number of di-jet events :  " << norm_dijet  <<endl;
    
  }
    
    
  private:
    
    AIDA::IHistogram1D *_hist_mb_09;
    AIDA::IHistogram1D *_hist_dijet_09;
    AIDA::IHistogram1D *_hist_mb_7;
    AIDA::IHistogram1D *_hist_dijet_7;
    
    
  };
  
  
  
  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_I930319> plugin_CMS_2011_I930319;
  
  

}
