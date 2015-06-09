#define l1macroexample_cxx
#include "L1MacroExample.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <iostream>
#include <TF1.h>

// some accessories

// Function to check for SA-MB2 to be within wedge
bool is_wedge(float mb2_phi, int sec){   // sectors 1..13
  Float_t pig=3.14159265; 
  Float_t sec_phi_width = pig/6.;  // phi width of one wheel in rads
  if (sec == 1) {
    return (mb2_phi< 0.5*sec_phi_width || mb2_phi > 11.5*sec_phi_width);
  }
  else {
    return   (mb2_phi > ((-1.5+sec)*sec_phi_width) &&
		 mb2_phi < ((-0.5+sec)*sec_phi_width));
  }
}

// Function to check for SA-MB2 to be within wheel
bool is_wheel(float mb2_z, int wheel){ // wheels -2..+2
  Float_t sec_z_width = 266.;     // z width of one wheel in cm
  return (mb2_z > ((-0.5+wheel)*sec_z_width) && mb2_z < ((+0.5+wheel)*sec_z_width));
}


//////////////////////////////////////////////////////////////////////
void L1MacroExample::Loop()
{
  if (fChain == 0)  return;

  Float_t pig=3.14159265; 

  // Muon cuts constants
  Float_t phi_sep_cut = 1.0;  // req'd separation in phi (rad) of 2 muons
  Float_t valid_hits_cut = 20.0; // red'd number of hits to accept SA
  Float_t pt_cut = 4.0;          // req'd minimum pt of the SA
  
  TFile fout("./L1MacroExample_output.root","recreate");
  TH1F* h1_DTTF_Phi_Resolution_all = 0;

  
  if (domuonreco){
    fout.mkdir("L1Muon");
    fout.cd("L1Muon");
    h1_DTTF_Phi_Resolution_all = 
      new TH1F("h1_DTTF_phi_res_all","DTTF resolution, all sectors",200,-0.3,0.3);
  }
  
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  unsigned int nevents =0;
  unsigned int nevents_=0;
 
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    nevents++;
    if (nevents>nevents_+10000){
      std::cout<<"Already processed "<<nevents/1000<<"k events"<<std::endl;
      nevents_=nevents;
    }

    if (domuonreco){
      Float_t phidiff = 1000.;
      if (nMuons==2) {
	phidiff = fabs(muons_sa_phi_mb2[0]-muons_sa_phi_mb2[1]);
	phidiff=(phidiff>pig)?(2*pig-phidiff):(phidiff);
      }
      if (nMuons==1 || (nMuons==2 && phidiff > phi_sep_cut)){
	for ( Int_t i_mu=0; i_mu < nMuons; i_mu++) { // loop over muon
	  bool accept_hitcut = (muons_sa_validhits[i_mu]>valid_hits_cut);
	  bool accept_ptcut = (muons_sa_pt[i_mu]>pt_cut);
	  if (accept_hitcut && accept_ptcut){
	    Float_t dphidt1=999; Int_t closedt1=999;
	    Float_t dphidt2=999; Int_t closedt2=999;
	    for (Int_t idt=0; idt<gmtNdt; idt++){   // loop on DTTF candidates
	      Float_t newsep = muons_sa_phi_mb2[i_mu]-gmtPhidt[idt]-(pig/144.);
	      if (newsep<-pig) newsep = newsep + 2*pig;
	      if (newsep> pig) newsep = newsep - 2*pig;
	      Float_t newsep_abs = fabs(newsep);
	      if (newsep_abs < fabs(dphidt1)){
		dphidt2 = dphidt1;
		dphidt1 = newsep;
		closedt2 = closedt1;
		closedt1 = idt;
	      }
	      else if (newsep_abs < fabs(dphidt2)){
		dphidt2=newsep;
		closedt2=idt;
	      }
	    } // end of loop on DTTF candidates, now triggers are matched
	    if (dphidt1<999 && muon_type[i_mu] == GL_MUON) { 
	      // here we have a match with a GLOBAL MUON
	      Float_t rsquare = 
		muons_imp_point_x[i_mu]*muons_imp_point_x[i_mu] +
		muons_imp_point_y[i_mu]*muons_imp_point_y[i_mu];
	      if (rsquare < 10*10 && fabs(muons_imp_point_z[i_mu])<50.){
		// these would pass the superpointing cut
		h1_DTTF_Phi_Resolution_all->Fill(dphidt1);
	      }
	    }
	  }
	}
      }
    } // end of muon block
  }  
  
  // Fits
  if (dofits){
    if (domuonreco){
      Double_t minfit_dphi=-0.2, maxfit_dphi=0.2;
      TF1 *gfit = new TF1("Gaussian","gaus",minfit_dphi,maxfit_dphi); 
      h1_DTTF_Phi_Resolution_all->Fit(gfit,"RQ");
    }
  }
  std::cout<<"Processed "<<nevents<<" events."<<std::endl;
  fout.Write();
  fout.Close();
   
}
