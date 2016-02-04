// -*- C++ -*- 
// Author: A. Knutsson
// Version: 0.3, 18/10-2010
// 	
// 18/10-2010: 
//	Particle loop for counting moved, no longer inside the main particle loop (just a time consuming issue)
//	Particle counting corrected:
//	    Events are multiplcity classified with respect to crhg particles with pt>0.4 
//	    Normalization factor based on mulpliplicities within pt-bin. 
//
// Note: You currently need to link ROOT when compiling. The histograms are
//	 standard ROOT histograms which are stored in a ROOT file. AIDA is not
//	 used.
//
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/Beam.hh"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"


namespace Rivet {

  class CMS_2010_2PCORRELATION : public Analysis {
  public:

    CMS_2010_2PCORRELATION() : Analysis("CMS_2010_2PCORRELATION") {
       setBeams(PROTON, PROTON);
       setNeedsCrossSection(false);
    }

//AK =====================================================INIT
    void init() {
      ChargedFinalState cfs(-2.4, 2.4, 0.1*GeV); //Note the eta selection for charged particles is here
      addProjection(cfs, "CFS");
      addProjection(Beam(), "Beam");
 	
	_N110events = 0;
	_Nevt_after_cuts = 0;
	
	 file = new TFile("cmsridge.root","recreate");
	
        for (int ibin = 0; ibin < 16; ibin++) {
	   
        Char_t hname[100];
        sprintf(hname,"S_DeltaPhi_%i",ibin+1);
            _h_S_DeltaPhi_Nptbins[ibin] = new TH1F(hname, hname,17,0.0,PI);
        sprintf(hname,"B_DeltaPhi_%i",ibin+1);
            _h_B_DeltaPhi_Nptbins[ibin] = new TH1F(hname, hname,17,0.0,PI);
	}

        _h2_B_dphivsdeta_mb_01pt = new TH2F("B_MB_01pt","B MB 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_S_dphivsdeta_mb_01pt = new TH2F("S_MB_01pt","S MB 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        
	_h2_S_dphivsdeta_N110_01pt = new TH2F("S_N110_01pt","S N>110 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_N110_01pt = new TH2F("B_N110_01pt","B N>110 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
       
        _h2_S_dphivsdeta_mb_1pt3 = new TH2F("S_MB_1pt3","S MB 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_mb_1pt3 = new TH2F("B_MB_1pt3","B MB 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);

        _h2_S_dphivsdeta_N110_1pt3 = new TH2F("S_N110_1pt3","S N>110 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_N110_1pt3 = new TH2F("B_N110_1pt3","B N>110 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
	
	 
	_hPhi_01pt = new TH1F("phi_pt01","phi 0.1<pt",100,-2*PI,2*PI);
    }

//AK =====================================================ANALYZE
    void analyze(const Event& event) {


      const double _ptbinslimits[5] = {0.1,1.0,2.0,3.0,4.0};
      const unsigned int _Nbinslimits[5] = {1, 35, 90, 110, 9999};

      const double weight = event.weight();
      const ChargedFinalState& charged = applyProjection<ChargedFinalState>(event, "CFS");
      const ParticleVector ChrgParticles = charged.particles();

      _Nevt_after_cuts++;
 
      //count particles for the signal event
      unsigned int Nparts_01pt=0;
      unsigned int Nparts_04pt=0;
      unsigned int Nparts_1pt3=0;
      unsigned int Nparts_ptbin[4]={0};
      foreach (const Particle& p, charged.particles()) {
        double pT = p.momentum().pT();
	if (pT>0.1){Nparts_01pt++;}
	if (pT>0.4){Nparts_04pt++;}
	for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	  if (pT > _ptbinslimits[iPtbin] && pT < _ptbinslimits[iPtbin+1]){	  	  	         
	     Nparts_ptbin[iPtbin]++;
	   }
	 } //end - pt-bin loop 	 
      }      	
      Nparts_1pt3 =  Nparts_ptbin[1] + Nparts_ptbin[2];
	

      //determine the multiplcity bin
      int Nbin=-99;
      for (int iNbin = 0; iNbin < 4; iNbin++) {
      	if(Nparts_04pt > _Nbinslimits[iNbin] && Nparts_04pt < _Nbinslimits[iNbin+1]) Nbin=iNbin; //Nbin now starts at 0
      }

      //particle count - MB background event
      int oldNpartsMB_01pt=0;
      int oldNpartsMB_1pt3=0;
      foreach (const Particle& p, _oldpartvecMB) {
        double pT = p.momentum().pT();
	if (pT>0.1){oldNpartsMB_01pt++;}
	if (pT>1.0 && pT<3.0){oldNpartsMB_1pt3++;}
      }      	
 
      //Get the background for the N classified event	
      ParticleVector oldpartvecNBin;
      if (Nbin == 0){ oldpartvecNBin = _oldpartvec1N35;}
      if (Nbin == 1){ oldpartvecNBin = _oldpartvec35N90;}
      if (Nbin == 2){ oldpartvecNBin = _oldpartvec90N110;}
      if (Nbin == 3){ oldpartvecNBin = _oldpartvec110N;} 
      
      //particle count for the N classified background event	
      int oldNparts_01pt=0;
      int oldNparts_1pt3=3;
      int oldNparts_ptbin[4]={0};
      foreach (const Particle& p, oldpartvecNBin)  {
         double pT = p.momentum().pT();
	 if(pT>0.1) oldNparts_01pt++;
	 for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	   if (pT > _ptbinslimits[iPtbin] && pT < _ptbinslimits[iPtbin+1]){	  	  	         
	    oldNparts_ptbin[iPtbin]++;
	   }
	 } //end - pt-bin loop 	 
       }
       oldNparts_1pt3 =  oldNparts_ptbin[1] + oldNparts_ptbin[2];
	

      if(oldpartvecNBin.size() != 0 && _oldpartvecMB.size() != 0) {  //only carry on with filling plots if we already have a one background event
      
      for (unsigned int ip1 = 0; ip1 < ChrgParticles.size(); ip1++) {
        const Particle& p1 = ChrgParticles[ip1];

        double pT1 = p1.momentum().pT();      	
        double eta1 = p1.momentum().eta();
        double phi1 = p1.momentum().phi();
        
        _hPhi_01pt->Fill(phi1, weight); //just test histo	  

	
	// Loop same event for S-distributions
       for (unsigned int ip2 = ip1+1; ip2 < ChrgParticles.size(); ip2++) {
        const Particle& p2 = ChrgParticles[ip2];
          
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = fabs(phi1-phi2);
		  	  
	  if(dphi >= PI) {dphi = 2*PI - dphi;}
	  
	  //1D (R vs DeltaPhi) - Signal
          if (deta > 2.0 && deta < 4.8){
	    for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	      if (pT1 > _ptbinslimits[iPtbin] && pT1 < _ptbinslimits[iPtbin+1] &&  pT2 >  _ptbinslimits[iPtbin] && pT2 <  _ptbinslimits[iPtbin+1]){	  	  	         
		   int ibin = iPtbin + Nbin*4;  //which histo to fill, 4x4
                   double pweight=1.0/(Nparts_ptbin[iPtbin]*(Nparts_ptbin[iPtbin]*-1));
                  _h_S_DeltaPhi_Nptbins[ibin]->Fill(dphi, pweight);	  	      	   
	      }
	    } //end - pt-bin loop 
   	  } //end - if 2.0 < deta < 4.8
	  
	  
	  //3D plots (R vs DeltaPhi vs DeltaEta) - Signal
	  if (pT1 >= 0.1 && pT2 >= 0.1){	  	  
              double pweight=1.0/(Nparts_01pt*(Nparts_01pt-1));	     
	      //MB (3D - S - low pt)
              _h2_S_dphivsdeta_mb_01pt->Fill(deta, dphi, pweight);	  
              _h2_S_dphivsdeta_mb_01pt->Fill(-deta, dphi, pweight);	  
	      if (dphi <= 0.5*PI){
                   _h2_S_dphivsdeta_mb_01pt->Fill(deta, -dphi, pweight);	  
                   _h2_S_dphivsdeta_mb_01pt->Fill(-deta, -dphi, pweight);	  
	      
	      }else{	      
                   _h2_S_dphivsdeta_mb_01pt->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_S_dphivsdeta_mb_01pt->Fill(-deta, (2.*PI-dphi), pweight);	  
	      }	   	      
	      //N>110 (3D - S - low pt)
	     if(Nparts_04pt>=110){
                _h2_S_dphivsdeta_N110_01pt->Fill(deta, dphi, pweight);	  
                _h2_S_dphivsdeta_N110_01pt->Fill(-deta, dphi, pweight);	  
	        if (dphi <= 0.5*PI){
                   _h2_S_dphivsdeta_N110_01pt->Fill(deta, -dphi, pweight);	  
                   _h2_S_dphivsdeta_N110_01pt->Fill(-deta, -dphi, pweight);	  
	      
	        }else{	      
                   _h2_S_dphivsdeta_N110_01pt->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_S_dphivsdeta_N110_01pt->Fill(-deta, (2.*PI-dphi), pweight);	  
	        }	      	        	     
	     }	//end - N>110 	     
	  } //end - pt cuts
	
	  	  
	  if (pT1 >= 1 && pT1 <= 3 && pT2 >= 1 && pT2 <= 3){ 
              double pweight=1.0/(Nparts_1pt3*(Nparts_1pt3-1));
	      //MB (3D - S  1<pt<3)
              _h2_S_dphivsdeta_mb_1pt3->Fill(deta, dphi, pweight);	  
              _h2_S_dphivsdeta_mb_1pt3->Fill(-deta, dphi, pweight);	  
	      if (dphi <= 0.5*PI){
                   _h2_S_dphivsdeta_mb_1pt3->Fill(deta, -dphi, pweight);	  
                   _h2_S_dphivsdeta_mb_1pt3->Fill(-deta, -dphi, pweight);	  
	      
	      }else{	      
                   _h2_S_dphivsdeta_mb_1pt3->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_S_dphivsdeta_mb_1pt3->Fill(-deta, (2.*PI-dphi), pweight);	  
	      }	     	      
	      //N>110 (3D - S  1<pt<3)
	     if(Nparts_04pt>=110){
                _h2_S_dphivsdeta_N110_1pt3->Fill(deta, dphi, pweight);	  
                _h2_S_dphivsdeta_N110_1pt3->Fill(-deta, dphi, pweight);	  
	        if (dphi <= 0.5*PI){
                   _h2_S_dphivsdeta_N110_1pt3->Fill(deta, -dphi, pweight);	  
                   _h2_S_dphivsdeta_N110_1pt3->Fill(-deta, -dphi, pweight);	  
	      
	        }else{	      
                   _h2_S_dphivsdeta_N110_1pt3->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_S_dphivsdeta_N110_1pt3->Fill(-deta, (2.*PI-dphi), pweight);	  
	        }	      
	     } //end - N>110 		     
	  }//end - pt cuts
	
		
      } //end - 2nd particle for the current event


////
///////Done with Signal - Now do the Backgrounds
////

      //Background bussiness MB
     
      if (_oldpartvecMB.size() > 0){ //only if background is already filled
       
      // Loop old MB event for B-distributions
      for (unsigned int ip2 = 0; ip2 < _oldpartvecMB.size(); ip2++) {
          const Particle& p2 = _oldpartvecMB[ip2];
	          
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = fabs(phi1-phi2);
		  	  				  
	  if(dphi >= PI) {dphi = 2*PI - dphi;}
	   	
	  //MB (3D - B - low pt)
	  if (pT1 >= 0.1 && pT2 >= 0.1){	  	  
              double pweight=1.0/(Nparts_01pt*oldNpartsMB_01pt);
              _h2_B_dphivsdeta_mb_01pt->Fill(deta, dphi, pweight);	  
              _h2_B_dphivsdeta_mb_01pt->Fill(-deta, dphi, pweight);	  
	      if (dphi <= 0.5*PI){
                   _h2_B_dphivsdeta_mb_01pt->Fill(deta, -dphi, pweight);	  
                   _h2_B_dphivsdeta_mb_01pt->Fill(-deta, -dphi, pweight);	  
	      
	      }else{	      
                   _h2_B_dphivsdeta_mb_01pt->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_B_dphivsdeta_mb_01pt->Fill(-deta, (2.*PI-dphi), pweight);	  
	      }	     	     		     
	  } //end - pt cuts

	  //MB (3D - B - 1<pt<3)
	  if (pT1 >= 1 && pT1 <= 3 && pT2 >= 1 && pT2 <= 3){ 
              double pweight=1.0/(Nparts_1pt3*oldNpartsMB_1pt3);
              _h2_B_dphivsdeta_mb_1pt3->Fill(deta, dphi, pweight);	  
              _h2_B_dphivsdeta_mb_1pt3->Fill(-deta, dphi, pweight);	  
	      if (dphi <= 0.5*PI){
                   _h2_B_dphivsdeta_mb_1pt3->Fill(deta, -dphi, pweight);	  
                   _h2_B_dphivsdeta_mb_1pt3->Fill(-deta, -dphi, pweight);	  
	      
	      }else{	      
                   _h2_B_dphivsdeta_mb_1pt3->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_B_dphivsdeta_mb_1pt3->Fill(-deta, (2.*PI-dphi), pweight);	  
	      }	     	      
	  }//end - pt cuts


        } //end - particle loop over saved MB particle vector

       } //end - need atleast 1 background event 
      
      
	//Background bussiness N>110
      if (_oldpartvec110N.size() > 100){ //only if it is already filled
      
      
      // Loop old N110 event for B-distributions
      for (unsigned int ip2 = 0; ip2 < _oldpartvec110N.size(); ip2++) {
          const Particle& p2 = _oldpartvec110N[ip2];
         
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = fabs(phi1-phi2);
		  	  				  
	  if(dphi >= PI) {dphi = 2*PI - dphi;}
	  
   	
	  if (pT1 >= 0.1 && pT2 >= 0.1){	  	  
	      //Fill 
              double pweight=1.0/(Nparts_01pt*oldNparts_01pt);
              _h2_B_dphivsdeta_N110_01pt->Fill(deta, dphi, pweight);	  
              _h2_B_dphivsdeta_N110_01pt->Fill(-deta, dphi, pweight);	  
	      if (dphi <= 0.5*PI){
                   _h2_B_dphivsdeta_N110_01pt->Fill(deta, -dphi, pweight);	  
                   _h2_B_dphivsdeta_N110_01pt->Fill(-deta, -dphi, pweight);	  
	      
	      }else{	      
                   _h2_B_dphivsdeta_N110_01pt->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_B_dphivsdeta_N110_01pt->Fill(-deta, (2.*PI-dphi), pweight);	  
	      }	     	     		     
	  } //end - pt cuts

	  if (pT1 >= 1 && pT1 <= 3 && pT2 >= 1 && pT2 <= 3){ 
	      //Fill 
              double pweight=1.0/(Nparts_1pt3*oldNparts_1pt3);
	      _h2_B_dphivsdeta_N110_1pt3->Fill(deta, dphi, pweight);	  
              _h2_B_dphivsdeta_N110_1pt3->Fill(-deta, dphi, pweight);	  
	      if (dphi <= 0.5*PI){
                   _h2_B_dphivsdeta_N110_1pt3->Fill(deta, -dphi, pweight);	  
                   _h2_B_dphivsdeta_N110_1pt3->Fill(-deta, -dphi, pweight);	  
	      
	      }else{	      
                   _h2_B_dphivsdeta_N110_1pt3->Fill(deta, (2.*PI-dphi), pweight);	  
                   _h2_B_dphivsdeta_N110_1pt3->Fill(-deta, (2.*PI-dphi), pweight);	  
	      }	     	      
	  }//end - pt cuts


         } //end - particle loop old N110
	
	} //end - need atleast 1 event already 



////
///////// Here we will do the 4x4 - 1D R - DeltaPhi Background
////	
      if (oldpartvecNBin.size() > _Nbinslimits[Nbin]){ //only if we already have BG particles            	
     
      // Loop old event for B-distributions
        for (unsigned int ip2 = 0; ip2 < oldpartvecNBin.size(); ip2++) {
          const Particle& p2 = oldpartvecNBin[ip2];
        
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = fabs(phi1-phi2);
		  	  				  
	  if(dphi >= PI) {dphi = 2*PI - dphi;}
	 
	  //loop the pt bins for the DeltaPhi - 1D - Background
          if (deta > 2.0 && deta < 4.8){
	    for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	      if (pT1 > _ptbinslimits[iPtbin] && pT1 < _ptbinslimits[iPtbin+1] &&  pT2 >  _ptbinslimits[iPtbin] && pT2 <  _ptbinslimits[iPtbin+1]){	  	  
		   int ibin = iPtbin + Nbin*4;  //which histo to fill, 4x4 matix
                   double pweight=1.0/(Nparts_ptbin[iPtbin]*oldNparts_ptbin[iPtbin]);
                  _h_B_DeltaPhi_Nptbins[ibin]->Fill(dphi, pweight);	  
	      } //end check pt-bin
	    } //end - pt-bin loop 
   	  } //end - if 2.0 < deta < 4.8
	  
   	
         } //end - particle loop old N110	
	} //end - need atleast 1 event already 
      

      } //end - main particle loop for current event    
    } //end - if background events found
	
	
//save the old particle vector
    if (Nbin == 0){ _oldpartvec1N35 = ChrgParticles;}  //TODO: would be nicer with an array or vec<>
    if (Nbin == 1){ _oldpartvec35N90 = ChrgParticles;}
    if (Nbin == 2){ _oldpartvec90N110 = ChrgParticles;}
    if (Nbin == 3){ _oldpartvec110N = ChrgParticles; _N110events++;}
    if (Nparts_01pt >= 1){ _oldpartvecMB = ChrgParticles;}
    
    } //end - analyze()
    
//AK =====================================================FINALIZE
    void finalize() {
    	double normfac=1.0/_Nevt_after_cuts;  
	cout << "Events:  " << _Nevt_after_cuts << "  " << normfac << endl;	
	
	file -> Write();
	
	//TODO: calc R already here (currently done in plot script)
	
	cout <<  "Number of events with N>110:" << _N110events << endl;
	
    }


//AK =====================================================DECLARATIONS
  private:

    double detamin;
    double detamax;	
    int Ndetabins;
    
    double _N110events;
    double _Nevt_after_cuts;


        TH1F *_hPhi_01pt;
        TH1F *_h_S_DeltaPhi_Nptbins[16];
        TH1F *_h_B_DeltaPhi_Nptbins[16];

        TH2F *_h2_S_dphivsdeta_mb_01pt;
        TH2F *_h2_S_dphivsdeta_mb_1pt3;

        TH2F *_h2_B_dphivsdeta_mb_01pt;
        TH2F *_h2_B_dphivsdeta_mb_1pt3;
	
        TH2F *_h2_S_dphivsdeta_N110_01pt;
        TH2F *_h2_S_dphivsdeta_N110_1pt3;

        TH2F *_h2_B_dphivsdeta_N110_01pt;
        TH2F *_h2_B_dphivsdeta_N110_1pt3;
	
	ParticleVector _oldpartvec1N35;
	ParticleVector _oldpartvec35N90;
	ParticleVector _oldpartvec90N110; //TODO: this is a bit ugly -> arrays or vec<vec<>>
	ParticleVector _oldpartvec110N;
	ParticleVector _oldpartvecMB;
	
	TFile *file;
   };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2010_2PCORRELATION> plugin_CMS_2010_2PCORRELATION;

}

