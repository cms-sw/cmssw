// -*- C++ -*-

// Samantha Dooling - Summer 2010, DESY
// Albert Knutsson (albert.knutsson@cern.ch) - some updates - Spring 2011, DESY

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


  class CMS_FWD_10_011 : public Analysis 
  {
  
  public:

    

    /// Constructor
    
    CMS_FWD_10_011()
      : Analysis("CMS_FWD_10_011")
      
      {
      	setBeams(PROTON,PROTON);
      }
      
 	double evcounter_dijet;  		//to count the number of selected events
	double evcounter_mb;  		//to count the number of selected events
 	double rejectedevents;
		
     	
	//counters 
	double cnt_eventsgenerated_noweight;
	double cnt_eventsgenerated;
	double cnt_eventspassingnd;
	double cnt_eventspassingptdijet;
	double cnt_eventspassingetadijet;
	double cnt_eventspassingdphidijet;
 
    /// Book histograms and initialise projections before the run
    
    void init() 
    {

    
      const FinalState fs(-6.0,6.0,0.0*GeV);             
      addProjection(fs, "FS"); 
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.5), "Jets");	

      VetoedFinalState fsv(fs);
      fsv.vetoNeutrinos();
      fsv.addVetoPairDetail(MUON, 0.0*GeV, 99999.9*GeV);
      addProjection(fsv, "fsv");



      // for the MB NSD selection
      const ChargedFinalState fschrgd(-6.0,6.0,0.0*GeV);             
      addProjection(fschrgd, "fschrgd"); 
      VetoedFinalState fschrgdv(fschrgd);
      fschrgdv.vetoNeutrinos();
      fschrgdv.addVetoPairDetail(MUON, 0.0*GeV, 99999.9*GeV);
      addProjection(fschrgdv, "fschrgdv");

      cnt_eventsgenerated = 0.0;
      cnt_eventsgenerated_noweight = 0.0;
      cnt_eventspassingnd = 0.0;
      cnt_eventspassingptdijet = 0.0;
      cnt_eventspassingetadijet = 0.0;
      cnt_eventspassingdphidijet = 0.0;
 
    
      evcounter_dijet =0;
      evcounter_mb =0;
      rejectedevents = 0;

	//just for control, no data comparison
      _hist_jets1 	= bookHistogram1D("pt1", 40, 0.5, 80.5); 	//jets with the highest pt
      _hist_jets2 	= bookHistogram1D("pt2", 40, 0.5, 80.5); 	//jets with the second highest pt
      _hist_eta1	= bookHistogram1D("eta1", 12, -3.0, 3.0); 	//eta distribution of the first jet (after the cuts)
      _hist_eta2	= bookHistogram1D("eta2", 12 ,-3.0, 3.0); 	//eta distribution of the second jet (after the cuts)
      _hist_dphi	= bookHistogram1D("dphi", 12, 0, 4 ); 		//delta phi distribution (before the cuts)
      _hist_dphi_cut	= bookHistogram1D("dphi_cut", 12, 0, 4); 	//delta phi distribution (after the cuts) -> delta peak at 180 degree
      _hist_num_p1	= bookHistogram1D("num_part1", 10, 0, 50); 	//number of particles in the first jet
      _hist_num_p2	= bookHistogram1D("num_part2", 10, 0, 50); 	//number of particles in the second jet
      _hist_num_j	= bookHistogram1D("num_jets", 10, 0, 10); 	//number of jets in the event
      _hist_num_jcut	= bookHistogram1D("num_jets_cut", 10, 0 , 10); 	//number of jets after the pt and eta cut        
      _hist_part_pt 	= bookHistogram1D("MB_pt_particles",  30, 0.0, 100.0); 	//jets with the highest pt
      _hist_part_E 	= bookHistogram1D("MB_E_particles",   30, 0.0, 1000.0); 	//jets with the highest pt
      _hist_part_eta 	= bookHistogram1D("MB_eta_particles", 30, -6.0, 6.0); 	//jets with the highest pt
      _hist_part_mul 	= bookHistogram1D("MB_mul_particles", 201, -0.5, 200.5); 	//jets with the highest pt
       
       
       //MAIN HISTOGRAMS for data comparisons
      if(fuzzyEquals(sqrtS(), 900, 1E-3)){
 	 _hist_CMS_FWD_10_011_mb_09 	 = bookHistogram1D(1,1,1); 	// energy flow in MB, 0.9 TeV
         _hist_CMS_FWD_10_011_dijet_09 	 = bookHistogram1D(2,1,1); 	// energy flow in dijet events, 0.9 TeV
      }
      if(fuzzyEquals(sqrtS(), 7000, 1E-3)){     
         _hist_CMS_FWD_10_011_mb_7 	 = bookHistogram1D(3,1,1); 	// energy flow in MB, 7 TeV
         _hist_CMS_FWD_10_011_dijet_7 	 = bookHistogram1D(4,1,1); 	// energy flow in dijet events, 7 TeV
	
      }
      
    }


    /// Perform the per-jet analysis
    void analyze(const Event& event) 
    {
    
     const double weight = event.weight();

    cnt_eventsgenerated += weight;
    cnt_eventsgenerated_noweight += 1.0;
     
     double diffphi;
                         
      	// Skip if the event is empty
       const FinalState& fsv = applyProjection<FinalState>(event, "fsv");
       if (fsv.empty()) 
       	{      	
         	vetoEvent;
       	}
	
	
      	// Veto diffraction according to defined hadron level.
	double count_chrg_forward=0;
	double count_chrg_backward=0;
        const FinalState& fschrgdv = applyProjection<FinalState>(event, "fschrgdv");
	foreach (const Particle& p, fschrgdv.particles())
	{
	   if( 3.9 < p.momentum().pseudorapidity() && 
	             p.momentum().pseudorapidity() < 4.4){count_chrg_forward++;}
	   if(-4.4 < p.momentum().pseudorapidity() && 
	             p.momentum().pseudorapidity() < -3.9){count_chrg_backward++;}
	 
	}	
        if(count_chrg_forward*count_chrg_backward==0) {
		rejectedevents++;
		vetoEvent;
		cout << "Event not rejected! Should be rejected! Check bug!"  << endl;}
	
	cnt_eventspassingnd += weight;
	
	
      const FastJets& jetpro = applyProjection<FastJets>(event, "Jets"); 
      const Jets& jets = jetpro.jetsByPt(7.0*GeV);
	
	    
//  XXXXXXXXXXXXXXXXXXXXXXXXXXXX MINIMUM BIAS EVENTS 					
	//loop over particles to calculate the energy
        _hist_part_mul->fill(fsv.particles().size(),1.0 );
	evcounter_mb += weight;
	foreach (const Particle& p, fsv.particles())
	{
		_hist_part_pt->fill(p.momentum().pT(),1.0 );
		_hist_part_eta->fill(p.momentum().pseudorapidity(),1.0 );
		_hist_part_E->fill(p.momentum().E(),1.0 );
		
		if(fuzzyEquals(sqrtS(), 900, 1E-3)) {_hist_CMS_FWD_10_011_mb_09 -> 
		  fill(fabs(p.momentum().pseudorapidity()), weight * p.momentum().E()/GeV );}
		
		if(fuzzyEquals(sqrtS(), 7000, 1E-3)) {_hist_CMS_FWD_10_011_mb_7 -> 
		  fill(fabs(p.momentum().pseudorapidity()), weight * p.momentum().E()/GeV );}
	
	}//foreach particle
	
    
    
//  XXXXXXXXXXXXXXXXXXXXXXXXXXXX DIJET EVENTS 
	

      _hist_num_j->fill(jets.size(), weight);	      
      
      if(jets.size()<2)
      {
      		
      }
      
      else
      {
 
	signed int index_1 = -1;	//for the jet with the 1.highest pT
	signed int index_2 = -1;	//for the jet with the 2.highest pT
		
	double tempmax_1 = -100;
	double tempmax_2 = -100;
	
	//find the jet with the 1.highest pt	
	for(signed int ijets = 0; ijets < (int)jets.size(); ijets++)
	{
	  if(tempmax_1 == -100 || tempmax_1 < jets[ijets].momentum().pT()/GeV)
	  {	
	    tempmax_1 = jets[ijets].momentum().pT()/GeV;
	    index_1 = ijets;
	  }		
	}
	
	
	
	//find the jet with the 2. highest pt	
	for(signed int ijets = 0; ijets < (int)jets.size(); ijets++)
	{		
	   if(tempmax_2 == -100 || tempmax_2 < jets[ijets].momentum().pT()/GeV)
	   {	
	     if(jets[ijets].momentum().pT()/GeV < tempmax_1)
	     {
		tempmax_2 = jets[ijets].momentum().pT()/GeV;
		index_2 = ijets;
             }
	   }				
	}
	
	
	if(index_1 != -1 && index_2 != -1)
	{
		
		diffphi = jets[index_2].momentum().phi() - jets[index_1].momentum().phi();    
				 
		if(diffphi < -PI)
		{	
			diffphi += 2.0*PI;  
		}
		if(diffphi > PI)
		{	
			diffphi -= 2.0*PI;  
		}
		diffphi = fabs(diffphi);
			
		_hist_dphi->fill(diffphi,weight);
		
	
	
		     
// ********************
// 900 GeV & 2360 GeV
// ********************

		
		//ask for the center of mass energy and do the pt cut
		if(fuzzyEquals(sqrtS(), 900, 1E-3))
		{	
			//pt cut
			if(jets[index_1].momentum().pT()/GeV > 8.0 && jets[index_2].momentum().pT()/GeV >8.0)
			{				
				cnt_eventspassingptdijet += weight;
	
				//eta cut for the central jets
				if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.5 && 
				   fabs(jets[index_2].momentum().pseudorapidity()) < 2.5)
				{
					
					cnt_eventspassingetadijet += weight;
      																						
					// fill the histogram with the number of Jets after the pt and eta cut 
					_hist_num_jcut->fill(jets.size(), weight);						
				 	
					//back to back condition of the jets
					if(fabs(diffphi-PI) < 1.0)
					{	
						cnt_eventspassingdphidijet += weight;
						evcounter_dijet += weight;
						
					
						//  fill varios jet distributions (for cross-checks, no data published) 						
						_hist_jets1->fill(jets[index_1].momentum().pT()/GeV ,weight);
						_hist_jets2->fill(jets[index_2].momentum().pT()/GeV ,weight);
						_hist_dphi_cut->fill(diffphi, weight);
						_hist_eta1->fill(jets[index_1].momentum().pseudorapidity() ,weight);
						_hist_eta2->fill(jets[index_2].momentum().pseudorapidity() ,weight);
						_hist_num_p1->fill(jets[index_1].size(), weight);
						_hist_num_p2->fill(jets[index_2].size(), weight);
						
																																
						//  E-flow 						
						foreach (const Particle& p, fsv.particles())
						{	
 					 	   _hist_CMS_FWD_10_011_dijet_09->
						     fill(fabs(p.momentum().pseudorapidity()), weight*p.momentum().E()/GeV);
						}//foreach particle
																																				 
					}//if(dphi)					
				}// else (eta cut central region)
			}//pt cut			
		}// energy request
		
		
		
// **********************		
// 7000 GeV		
		
		
		//ask for the center of mass energy and do the pt cut
		if(fuzzyEquals(sqrtS(), 7000, 1E-3))   
		{					
			//pt cut
			if(jets[index_1].momentum().pT()/GeV > 20.0 && jets[index_2].momentum().pT()/GeV > 20.0)
			{	
				cnt_eventspassingptdijet += weight;
			
				//eta cut for the central jets
				if(fabs(jets[index_1].momentum().pseudorapidity()) < 2.5 && 
				   fabs(jets[index_2].momentum().pseudorapidity()) < 2.5)
				{
      				
					cnt_eventspassingetadijet += weight;
					_hist_num_jcut->fill(jets.size(), weight);
						
					//back to back condition of the jets
					if(fabs(diffphi-PI) < 1.0)
					{	
						evcounter_dijet += weight;
						cnt_eventspassingdphidijet += weight;
						
			
 						//  fill varios jet distributions (for cross-checks, no data published) 
						_hist_jets1->fill(jets[index_1].momentum().pT()/GeV ,weight);
						_hist_jets2->fill(jets[index_2].momentum().pT()/GeV ,weight);
						_hist_dphi_cut->fill(diffphi, weight);
						_hist_eta1->fill(jets[index_1].momentum().pseudorapidity() ,weight);
						_hist_eta2->fill(jets[index_2].momentum().pseudorapidity() ,weight);
						_hist_num_p1->fill(jets[index_1].size(), weight);
						_hist_num_p2->fill(jets[index_2].size(), weight);
						
						
						//E-flow																								
						foreach (const Particle& p, fsv.particles())
						{								
						   _hist_CMS_FWD_10_011_dijet_7->
						     fill(fabs(p.momentum().pseudorapidity()), weight*p.momentum().E()/GeV);
						}//foreach particle
					}//if(dphi)
				}// else (eta cut central region)
			}//pt cut
		}// energy 				
	}// if index 
		
	
}// analysis

}

    /// Normalise histograms etc., after the run
    
    
    void finalize() 
    {
      
      const double sclfactor = 1.0/sumOfWeights();
      scale(_hist_jets1, sclfactor);
      scale(_hist_jets2, sclfactor);
      scale(_hist_eta1, sclfactor);
      scale(_hist_eta2, sclfactor);
      scale(_hist_dphi, sclfactor);
      scale(_hist_dphi_cut, sclfactor);
      scale(_hist_num_p1, sclfactor);
      scale(_hist_num_p2, sclfactor);
      scale(_hist_num_j, sclfactor);
      scale(_hist_num_jcut, sclfactor);
      
      
      const double norm_dijet =evcounter_dijet*2.0 ; //AK norm factor 2 for the +/- region
      const double norm_mb =evcounter_mb*2.0 ;
      	    
      if(fuzzyEquals(sqrtS(), 900, 1E-3)){
         scale(_hist_CMS_FWD_10_011_mb_09, 1.0/norm_mb);
         scale(_hist_CMS_FWD_10_011_dijet_09, 1.0/norm_dijet);
      }
      if(fuzzyEquals(sqrtS(), 7000, 1E-3)){
        scale(_hist_CMS_FWD_10_011_dijet_7, 1.0/norm_dijet);
        scale(_hist_CMS_FWD_10_011_mb_7, 1.0/norm_mb);
      }	       
     
     getLog() << Log::INFO << " " << endl;
     getLog() << Log::INFO << " " << endl;
     getLog() << Log::INFO << " " << endl;
     getLog() << Log::INFO << "Efficient number of events generated (weight=1.0):  " << cnt_eventsgenerated_noweight << "        (Average weight:" << cnt_eventsgenerated/cnt_eventsgenerated_noweight << ")" << endl;
     getLog() << Log::INFO << "Number of events generated:  " << cnt_eventsgenerated << "        (" << 100.0*cnt_eventsgenerated/cnt_eventsgenerated << "%)" <<endl;
     getLog() << Log::INFO << "Number of events passing ND selection:  " << cnt_eventspassingnd <<  "        (" << 100.0*cnt_eventspassingnd/cnt_eventsgenerated << "%)" <<endl;
     getLog() << Log::INFO << "Number of events passing ND + jet-Pt cut:  " << cnt_eventspassingptdijet <<  "        (" << 100.0*cnt_eventspassingptdijet/cnt_eventsgenerated << "%)" <<endl;
     getLog() << Log::INFO << "Number of events passing ND + jet-Pt + jet-eta:  " <<cnt_eventspassingetadijet  <<  "        (" << 100.0*cnt_eventspassingetadijet/cnt_eventsgenerated << "%)" <<endl;
     getLog() << Log::INFO << "Number of events passing ND + jet-Pt + jet-eta + DeltaPhi cut:  " << cnt_eventspassingdphidijet <<  "        (" << 100.0*cnt_eventspassingdphidijet/cnt_eventsgenerated << "%)" <<endl;
      
    }

    
  private:

    AIDA::IHistogram1D *_hist_part_pt;
    AIDA::IHistogram1D *_hist_part_E;
    AIDA::IHistogram1D *_hist_part_eta;
    AIDA::IHistogram1D *_hist_part_mul;
   
    AIDA::IHistogram1D *_hist_jets1;
    AIDA::IHistogram1D *_hist_jets2;
    AIDA::IHistogram1D *_hist_eta1;
    AIDA::IHistogram1D *_hist_eta2;
    AIDA::IHistogram1D *_hist_dphi;
    AIDA::IHistogram1D *_hist_dphi_cut;
    AIDA::IHistogram1D *_hist_num_p1;
    AIDA::IHistogram1D *_hist_num_p2;
    AIDA::IHistogram1D *_hist_num_j;
    AIDA::IHistogram1D *_hist_num_jcut;
    
    AIDA::IHistogram1D *_hist_CMS_FWD_10_011_mb_09;
    AIDA::IHistogram1D *_hist_CMS_FWD_10_011_dijet_09;
    AIDA::IHistogram1D *_hist_CMS_FWD_10_011_mb_7;
    AIDA::IHistogram1D *_hist_CMS_FWD_10_011_dijet_7;
    
 
  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_FWD_10_011> plugin_CMS_FWD_10_011;


}
