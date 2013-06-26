// -*- C++ -*- 
// Author: A. Knutsson
// Version: 1.0, 10/11-2010
//		
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


#include <TH1D.h>
#include <TH2D.h>
#include <TNtuple.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>
#include <TCanvas.h>
#include <TVector3.h>
#include <TRandom.h>

namespace Rivet {

  class CMS_2010_S8808686 : public Analysis {
  public:

    CMS_2010_S8808686() : Analysis("CMS_2010_S8808686") {
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
	
	 file = new TFile("CMS_2010_S8808686.root","recreate");
	
        for (int ibin = 0; ibin < 16; ibin++) {
	   
          Char_t hname[100];
        
	  sprintf(hname,"S_DeltaPhi_%i",ibin+1);
          _h_S_DeltaPhi_Nptbins[ibin] = new TH1F(hname, hname,17,0.0,PI);
        
	  sprintf(hname,"B_DeltaPhi_%i",ibin+1);
          _h_B_DeltaPhi_Nptbins[ibin] = new TH1F(hname, hname,17,0.0,PI);
        
	  sprintf(hname,"R_DeltaPhi_%i",ibin+1);
          _h_R_DeltaPhi_Nptbins[ibin] = new TH1F(hname, hname,17,0.0,PI);
	
	  sprintf(hname,"S_3D_Nptbin_%i",ibin+1);
          _h_S_3D_Nptbins[ibin] = new TH2F(hname,hname,24,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        
	  sprintf(hname,"B_3D_Nptbin_%i",ibin+1);
          _h_B_3D_Nptbins[ibin] = new TH2F(hname,hname,24,-4.8,4.8,30,-0.5*PI,3./2.*PI);
		
	  sprintf(hname,"R_3D_Nptbin_%i",ibin+1);
          _h_R_3D_Nptbins[ibin] = new TH2F(hname,hname,24,-4.8,4.8,30,-0.5*PI,3./2.*PI);

	  sprintf(hname,"mult_%i",ibin+1);
	  _hMult_Nptbins[ibin] = new  TH1F(hname,"N",250,0,250); 
		
	}

        _h2_S_dphivsdeta_mb_01pt = new TH2F("S_MB_01pt","S MB 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_mb_01pt = new TH2F("B_MB_01pt","B MB 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_R_dphivsdeta_mb_01pt = new TH2F("R_MB_01pt","R MB 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
	_hMult_mb_01pt = new  TH1F("mult_mb_01pt","N",250,0,250); 
        
	_h2_S_dphivsdeta_N110_01pt = new TH2F("S_N110_01pt","S N>110 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_N110_01pt = new TH2F("B_N110_01pt","B N>110 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_R_dphivsdeta_N110_01pt = new TH2F("R_N110_01pt","R N>110 0.1<pt",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
	_hMult_N110_01pt = new  TH1F("mult_N110_01pt","N",250,0,250); 
       
        _h2_S_dphivsdeta_mb_1pt3 = new TH2F("S_MB_1pt3","S MB 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_mb_1pt3 = new TH2F("B_MB_1pt3","B MB 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_R_dphivsdeta_mb_1pt3 = new TH2F("R_MB_1pt3","R MB 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
	_hMult_mb_1pt3 = new  TH1F("mult_mb_1pt3","N",250,0,250); 

        _h2_S_dphivsdeta_N110_1pt3 = new TH2F("S_N110_1pt3","S N>110 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_B_dphivsdeta_N110_1pt3 = new TH2F("B_N110_1pt3","B N>110 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
        _h2_R_dphivsdeta_N110_1pt3 = new TH2F("R_N110_1pt3","R N>110 1<pt<3",33,-4.8,4.8,30,-0.5*PI,3./2.*PI);
	_hMult_N110_1pt3 = new  TH1F("mult_N110_1pt3","N",250,0,250); 
	
	_hPhi_01pt = new TH1F("phi_pt01","phi 0.1<pt",100,-2*PI,2*PI);
    }

//AK =====================================================ANALYZE
    void analyze(const Event& event) {
	
      const double _ptbinslimits[5] = {0.1,1.0,2.0,3.0,4.0};
      const unsigned int _Nbinslimits[5] = {1, 35, 90, 110, 9999};

      //const double weight = event.weight();
      const ChargedFinalState& charged = applyProjection<ChargedFinalState>(event, "CFS");
      const ParticleVector ChrgParticles = charged.particles();
	
      if (ChrgParticles.size() <= 1) vetoEvent;

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
            
      //determine the multiplcity bin and fill particle multiplcity in pt bins
      int Nbin=-99;
      for (int iNbin = 0; iNbin < 4; iNbin++) {
      	if(Nparts_04pt > _Nbinslimits[iNbin] && Nparts_04pt < _Nbinslimits[iNbin+1]) {
	    Nbin=iNbin; //Nbin now starts at 0
	    for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
              _hMult_Nptbins[iPtbin + Nbin*4]->Fill(Nparts_ptbin[iPtbin]);
            }	    
	 }    
       }
       
       
       _hMult_mb_01pt->Fill(Nparts_01pt);
       _hMult_mb_1pt3->Fill(Nparts_1pt3);
       if (Nbin == 3) {
          _hMult_N110_01pt->Fill(Nparts_01pt);
          _hMult_N110_1pt3->Fill(Nparts_1pt3);
       } 
       
       
       //particle count - MB background event
      unsigned int oldNpartsMB_01pt=0;
      unsigned int oldNpartsMB_1pt3=0;
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
      unsigned int oldNparts_01pt=0;
      unsigned int oldNparts_1pt3=0;
      unsigned int oldNparts_ptbin[4]={0};
      foreach (const Particle& p, oldpartvecNBin)  {
         double pT = p.momentum().pT();
	 if(pT>0.1) oldNparts_01pt++;
	 for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	   if (pT > _ptbinslimits[iPtbin] && pT <= _ptbinslimits[iPtbin+1]){	  	  	         
	    oldNparts_ptbin[iPtbin]++;
	   }
	 } //end - pt-bin loop 	 
       }
       oldNparts_1pt3 =  oldNparts_ptbin[1] + oldNparts_ptbin[2];
	

      if(oldpartvecNBin.size() > _Nbinslimits[Nbin] && _oldpartvecMB.size() > 1 ) {  //only carry on with filling plots if we already have a one background event
      
      
      
      for (unsigned int ip1 = 0; ip1 < ChrgParticles.size(); ip1++) {
        const Particle& p1 = ChrgParticles[ip1];

        double pT1 = p1.momentum().pT();      	
        double eta1 = p1.momentum().eta();
        double phi1 = p1.momentum().phi();
        
        _hPhi_01pt->Fill(phi1, 1.0); //just test histo	  

	
	// Loop same event for S-distributions
       for (unsigned int ip2 = ip1+1; ip2 < ChrgParticles.size(); ip2++) {
        const Particle& p2 = ChrgParticles[ip2];
          
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = phi1-phi2;
		  	  
        if(dphi>PI) dphi=dphi-2*PI;
        if(dphi<-PI) dphi=dphi+2*PI;
	  
	  //1D (R vs DeltaPhi) - Signal
	    for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	      if (pT1 > _ptbinslimits[iPtbin] && pT1 < _ptbinslimits[iPtbin+1] &&  pT2 >  _ptbinslimits[iPtbin] && pT2 <  _ptbinslimits[iPtbin+1]){	  	  	         
		int ibin = iPtbin + Nbin*4;  //which histo to fill, 4x4
        	_h_S_3D_Nptbins[ibin]->Fill(fabs(deta),fabs(dphi),1.0/4.0/Nparts_ptbin[iPtbin]);
        	_h_S_3D_Nptbins[ibin]->Fill(-fabs(deta),fabs(dphi),1.0/4.0/Nparts_ptbin[iPtbin]);
        	_h_S_3D_Nptbins[ibin]->Fill(fabs(deta),-fabs(dphi),1.0/4.0/Nparts_ptbin[iPtbin]);
        	_h_S_3D_Nptbins[ibin]->Fill(-fabs(deta),-fabs(dphi),1.0/4.0/Nparts_ptbin[iPtbin]);
        	_h_S_3D_Nptbins[ibin]->Fill(fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_ptbin[iPtbin]);
        	_h_S_3D_Nptbins[ibin]->Fill(-fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_ptbin[iPtbin]);
		
		if(deta > 2.0 && deta < 4.8){
        	  _h_S_DeltaPhi_Nptbins[ibin]->Fill(fabs(dphi),1.0/Nparts_ptbin[iPtbin]);
		}
		
	      }
	    } //end - pt-bin loop 
	  
	  
	  //3D plots (R vs DeltaPhi vs DeltaEta) - Signal
	  if (pT1 >= 0.1 && pT2 >= 0.1){	  	  
        	_h2_S_dphivsdeta_mb_01pt->Fill(fabs(deta),fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_mb_01pt->Fill(-fabs(deta),fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_mb_01pt->Fill(fabs(deta),-fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_mb_01pt->Fill(-fabs(deta),-fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_mb_01pt->Fill(fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_mb_01pt->Fill(-fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_01pt);

	      //N>110 (3D - S - low pt)
	     if(Nparts_04pt>=110){
        	_h2_S_dphivsdeta_N110_01pt->Fill(fabs(deta),fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_N110_01pt->Fill(-fabs(deta),fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_N110_01pt->Fill(fabs(deta),-fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_N110_01pt->Fill(-fabs(deta),-fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_N110_01pt->Fill(fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_01pt);
        	_h2_S_dphivsdeta_N110_01pt->Fill(-fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_01pt);
	     }	//end - N>110 	     
	  } //end - pt cuts
	
	  	  
	  if (pT1 >= 1 && pT1 <= 3 && pT2 >= 1 && pT2 <= 3){ 
        	_h2_S_dphivsdeta_mb_1pt3->Fill(fabs(deta),fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_mb_1pt3->Fill(-fabs(deta),fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_mb_1pt3->Fill(fabs(deta),-fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_mb_1pt3->Fill(-fabs(deta),-fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_mb_1pt3->Fill(fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_mb_1pt3->Fill(-fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_1pt3);
	      //N>110 (3D - S  1<pt<3)
	     if(Nparts_04pt>=110){
       		_h2_S_dphivsdeta_N110_1pt3->Fill(fabs(deta),fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_N110_1pt3->Fill(-fabs(deta),fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_N110_1pt3->Fill(fabs(deta),-fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_N110_1pt3->Fill(-fabs(deta),-fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_N110_1pt3->Fill(fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_1pt3);
        	_h2_S_dphivsdeta_N110_1pt3->Fill(-fabs(deta),2*PI-fabs(dphi),1.0/4.0/Nparts_1pt3);
	     } //end - N>110 		     
	  }//end - pt cuts
	
		
      } //end - 2nd particle for the current event


////
///////Done with Signal - Now do the Backgrounds
////

      //Background bussiness MB
     
      if (_oldpartvecMB.size() > 1){ //only if background is already filled
       
      // Loop old MB event for B-distributions
      for (unsigned int ip2 = 0; ip2 < _oldpartvecMB.size(); ip2++) {
          const Particle& p2 = _oldpartvecMB[ip2];
	          
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = phi1-phi2;
		  	  				  
        if(dphi>PI) dphi=dphi-2*PI;
        if(dphi<-PI) dphi=dphi+2*PI;
	  
	   	
	  //MB (3D - B - low pt)
	  if (pT1 >= 0.1 && pT2 >= 0.1){	  	  
              double pweight=1.0/(Nparts_01pt*oldNpartsMB_01pt);
             _h2_B_dphivsdeta_mb_01pt->Fill(deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_01pt->Fill(-deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_01pt->Fill(deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_01pt->Fill(-deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_01pt->Fill(deta,2*PI-fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_01pt->Fill(-deta,2*PI-fabs(dphi),pweight);
 	  } //end - pt cuts

	  //MB (3D - B - 1<pt<3)
	  if (pT1 >= 1 && pT1 <= 3 && pT2 >= 1 && pT2 <= 3){ 
              double pweight=1.0/(Nparts_1pt3*oldNpartsMB_1pt3);
             _h2_B_dphivsdeta_mb_1pt3->Fill(deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_1pt3->Fill(-deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_1pt3->Fill(deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_1pt3->Fill(-deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_1pt3->Fill(deta,2*PI-fabs(dphi),pweight);
             _h2_B_dphivsdeta_mb_1pt3->Fill(-deta,2*PI-fabs(dphi),pweight);
	  }//end - pt cuts


        } //end - particle loop over saved MB particle vector

       } //end - need atleast 1 background event 
      
      
	//Background bussiness N>110
	
      if (oldpartvecNBin.size() > 100 &&  Nbin==3){ //only if it is already filled
      
      // Loop old N110 event for B-distributions
      for (unsigned int ip2 = 0; ip2 < oldpartvecNBin.size(); ip2++) {
          const Particle& p2 = oldpartvecNBin[ip2];
         
	  double pT2 = p2.momentum().pT();      	
          double eta2 = p2.momentum().eta();
          double phi2 = p2.momentum().phi();
        
	  double deta = fabs(eta1-eta2);
	  double dphi = phi1-phi2;
		  	  				  
        if(dphi>PI) dphi=dphi-2*PI;
        if(dphi<-PI) dphi=dphi+2*PI;
//        if(dphi>-PI && dphi<-PI/2.) dphi=dphi+2*PI;
	  
   	
	  if (pT1 >= 0.1 && pT2 >= 0.1){	  	  
	      //Fill 
              double pweight=1.0/(Nparts_01pt*oldNparts_01pt);
             _h2_B_dphivsdeta_N110_01pt->Fill(deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_01pt->Fill(-deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_01pt->Fill(deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_01pt->Fill(-deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_01pt->Fill(deta,2*PI-fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_01pt->Fill(-deta,2*PI-fabs(dphi),pweight);
 	  } //end - pt cuts

	  if (pT1 >= 1 && pT1 <= 3 && pT2 >= 1 && pT2 <= 3){ 
	      //Fill 
              double pweight=1.0/(Nparts_1pt3*oldNparts_1pt3);
             _h2_B_dphivsdeta_N110_1pt3->Fill(deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_1pt3->Fill(-deta,fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_1pt3->Fill(deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_1pt3->Fill(-deta,-fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_1pt3->Fill(deta,2*PI-fabs(dphi),pweight);
             _h2_B_dphivsdeta_N110_1pt3->Fill(-deta,2*PI-fabs(dphi),pweight);
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
	  double dphi = phi1-phi2;
		  	  				  
        if(dphi>PI) dphi=dphi-2*PI;
        if(dphi<-PI) dphi=dphi+2*PI;
	 
	  //loop the pt bins for the DeltaPhi - 1D - Background
	    for (int iPtbin = 0; iPtbin < 4; iPtbin++) {
	      if (pT1 > _ptbinslimits[iPtbin] && pT1 < _ptbinslimits[iPtbin+1] &&  pT2 >  _ptbinslimits[iPtbin] && pT2 <  _ptbinslimits[iPtbin+1]){	  	  
		   int ibin = iPtbin + Nbin*4;  //which histo to fill, 4x4 matix
                   double pweight=1.0/(Nparts_ptbin[iPtbin]*oldNparts_ptbin[iPtbin]);
             	_h_B_3D_Nptbins[ibin]->Fill(deta,fabs(dphi),pweight);
             	_h_B_3D_Nptbins[ibin]->Fill(-deta,fabs(dphi),pweight);
             	_h_B_3D_Nptbins[ibin]->Fill(deta,-fabs(dphi),pweight);
             	_h_B_3D_Nptbins[ibin]->Fill(-deta,-fabs(dphi),pweight);
             	_h_B_3D_Nptbins[ibin]->Fill(deta,2*PI-fabs(dphi),pweight);
             	_h_B_3D_Nptbins[ibin]->Fill(-deta,2*PI-fabs(dphi),pweight);
		if(deta > 2.0 && deta < 4.8){
        	  _h_B_DeltaPhi_Nptbins[ibin]->Fill(fabs(dphi),pweight);
		}
	      } //end check pt-bin
	    } //end - pt-bin loop 
	  
   	
         } //end - particle loop old N110	
	} //end - need atleast 1 event already 
      

      } //end - main particle loop for current event    
    } //end - if background events found
	
	
//save the old particle vector
    if (Nbin == 0){ _oldpartvec1N35 = ChrgParticles;}  
    if (Nbin == 1){ _oldpartvec35N90 = ChrgParticles;}
    if (Nbin == 2){ _oldpartvec90N110 = ChrgParticles;}
    if (Nbin == 3){ _oldpartvec110N = ChrgParticles; _N110events++;}
    if (Nparts_01pt >= 1){ _oldpartvecMB = ChrgParticles;}
    
    } //end - analyze()
    
//AK =====================================================FINALIZE
    void finalize() {

	getLog() << Log::INFO << "Number of events after event selection: " << _Nevt_after_cuts << endl;	
	getLog() << Log::INFO << "Number of events with N>110:" << _N110events << endl;
			
  	int nEvent = -99.0;
	
	nEvent = _hMult_mb_01pt->Integral(3,10000);
	   _h2_S_dphivsdeta_mb_01pt->Scale(1.0/nEvent);
	   _h2_B_dphivsdeta_mb_01pt->Scale(_h2_S_dphivsdeta_mb_01pt->Integral()/_h2_B_dphivsdeta_mb_01pt->Integral());
  	   _h2_R_dphivsdeta_mb_01pt->Add((TH2F*)_h2_S_dphivsdeta_mb_01pt);
 	   _h2_R_dphivsdeta_mb_01pt->Add(_h2_B_dphivsdeta_mb_01pt,-1);
  	   _h2_R_dphivsdeta_mb_01pt->Divide(_h2_B_dphivsdeta_mb_01pt);
  	   _h2_R_dphivsdeta_mb_01pt->Scale(_h2_S_dphivsdeta_mb_01pt->Integral());

  	 nEvent = _hMult_N110_01pt->Integral(3,10000);
	   _h2_S_dphivsdeta_N110_01pt->Scale(1.0/nEvent);
	   _h2_B_dphivsdeta_N110_01pt->Scale(_h2_S_dphivsdeta_N110_01pt->Integral()/_h2_B_dphivsdeta_N110_01pt->Integral());
  	   _h2_R_dphivsdeta_N110_01pt->Add(_h2_S_dphivsdeta_N110_01pt);
 	   _h2_R_dphivsdeta_N110_01pt->Add(_h2_B_dphivsdeta_N110_01pt,-1);
  	   _h2_R_dphivsdeta_N110_01pt->Divide(_h2_B_dphivsdeta_N110_01pt);
  	   _h2_R_dphivsdeta_N110_01pt->Scale(_h2_S_dphivsdeta_N110_01pt->Integral());

  	 nEvent = _hMult_mb_1pt3->Integral(3,10000);
	   _h2_S_dphivsdeta_mb_1pt3->Scale(1.0/nEvent);
	   _h2_B_dphivsdeta_mb_1pt3->Scale(_h2_S_dphivsdeta_mb_1pt3->Integral()/_h2_B_dphivsdeta_mb_1pt3->Integral());
  	   _h2_R_dphivsdeta_mb_1pt3->Add(_h2_S_dphivsdeta_mb_1pt3);
 	   _h2_R_dphivsdeta_mb_1pt3->Add(_h2_B_dphivsdeta_mb_1pt3,-1);
  	   _h2_R_dphivsdeta_mb_1pt3->Divide(_h2_B_dphivsdeta_mb_1pt3);
  	   _h2_R_dphivsdeta_mb_1pt3->Scale(_h2_S_dphivsdeta_mb_1pt3->Integral());

  	 nEvent = _hMult_N110_1pt3->Integral(3,10000);
	   _h2_S_dphivsdeta_N110_1pt3->Scale(1.0/nEvent);
	   _h2_B_dphivsdeta_N110_1pt3->Scale(_h2_S_dphivsdeta_N110_1pt3->Integral()/_h2_B_dphivsdeta_N110_1pt3->Integral());
  	   _h2_R_dphivsdeta_N110_1pt3->Add(_h2_S_dphivsdeta_N110_1pt3);
 	   _h2_R_dphivsdeta_N110_1pt3->Add(_h2_B_dphivsdeta_N110_1pt3,-1);
  	   _h2_R_dphivsdeta_N110_1pt3->Divide(_h2_B_dphivsdeta_N110_1pt3);
  	   _h2_R_dphivsdeta_N110_1pt3->Scale(_h2_S_dphivsdeta_N110_1pt3->Integral());

        for (int ibin = 0; ibin < 16; ibin++) {
  	   
	   int nEvent = _hMult_Nptbins[ibin]->Integral(3,10000);

	   _h_S_3D_Nptbins[ibin]->Scale(1.0/nEvent);
	   _h_B_3D_Nptbins[ibin]->Scale(_h_S_3D_Nptbins[ibin]->Integral()/_h_B_3D_Nptbins[ibin]->Integral());
  	   _h_R_3D_Nptbins[ibin]->Add(_h_S_3D_Nptbins[ibin]);
 	   _h_R_3D_Nptbins[ibin]->Add(_h_B_3D_Nptbins[ibin],-1);
  	   _h_R_3D_Nptbins[ibin]->Divide(_h_B_3D_Nptbins[ibin]);
  	   _h_R_3D_Nptbins[ibin]->Scale(_h_S_3D_Nptbins[ibin]->Integral());


   	    _h_S_DeltaPhi_Nptbins[ibin]->Scale(1.0/nEvent);    	    
	    // Wei's correction - to get the average multiplicity correct --
	    _hMult_Nptbins[ibin]->SetAxisRange(2,10000,"X");
     	    double rescale = (_hMult_Nptbins[ibin]->GetMean()-1)/_h_S_DeltaPhi_Nptbins[ibin]->Integral();
    	    _h_S_DeltaPhi_Nptbins[ibin]->Scale(rescale);
    	   //---------------------------------------------------------------
	   _h_B_DeltaPhi_Nptbins[ibin]->Scale(_h_S_DeltaPhi_Nptbins[ibin]->Integral()/_h_B_DeltaPhi_Nptbins[ibin]->Integral());
  	   _h_R_DeltaPhi_Nptbins[ibin]->Add(_h_S_DeltaPhi_Nptbins[ibin]);
 	   _h_R_DeltaPhi_Nptbins[ibin]->Add(_h_B_DeltaPhi_Nptbins[ibin],-1);
  	   _h_R_DeltaPhi_Nptbins[ibin]->Divide(_h_B_DeltaPhi_Nptbins[ibin]);
  	   _h_R_DeltaPhi_Nptbins[ibin]->Scale(_h_S_DeltaPhi_Nptbins[ibin]->Integral());
 

	}

	file -> Write();
	
    }


//AK =====================================================DECLARATIONS
  private:

    double detamin;
    double detamax;	
    int Ndetabins;
    
    double _N110events;
    double _Nevt_after_cuts;


        TH1F *_hMult;
        TH1F *_hPhi_01pt;
        TH1F *_h_S_DeltaPhi_Nptbins[16];
        TH1F *_h_B_DeltaPhi_Nptbins[16];
        TH1F *_h_R_DeltaPhi_Nptbins[16];

	TH2F *_h_S_3D_Nptbins[16];
	TH2F *_h_B_3D_Nptbins[16];
	TH2F *_h_R_3D_Nptbins[16];
        TH1F *_hMult_Nptbins[16];
	
        TH2F *_h2_S_dphivsdeta_mb_01pt;
        TH2F *_h2_B_dphivsdeta_mb_01pt;
        TH2F *_h2_R_dphivsdeta_mb_01pt;
        TH1F *_hMult_mb_01pt;
		
	TH2F *_h2_S_dphivsdeta_mb_1pt3;
        TH2F *_h2_B_dphivsdeta_mb_1pt3;
        TH2F *_h2_R_dphivsdeta_mb_1pt3;
	TH1F *_hMult_mb_1pt3;
	
        TH2F *_h2_S_dphivsdeta_N110_01pt;
        TH2F *_h2_B_dphivsdeta_N110_01pt;
        TH2F *_h2_R_dphivsdeta_N110_01pt;
	TH1F *_hMult_N110_01pt;
	
        TH2F *_h2_S_dphivsdeta_N110_1pt3;
        TH2F *_h2_B_dphivsdeta_N110_1pt3;	
        TH2F *_h2_R_dphivsdeta_N110_1pt3;
	TH1F *_hMult_N110_1pt3;
	
	ParticleVector _oldpartvec1N35;
	ParticleVector _oldpartvec35N90;
	ParticleVector _oldpartvec90N110; //TODO: this is a bit ugly -> arrays or vec<vec<>>
	ParticleVector _oldpartvec110N;
	ParticleVector _oldpartvecMB;
	
	TFile *file;
   };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2010_S8808686> plugin_CMS_2010_S8808686;

}

