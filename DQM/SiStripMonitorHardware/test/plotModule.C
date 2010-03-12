#include "iostream"
#include <vector>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TString.h"
#include "TLatex.h"


void plotModule(){//main

  //TFile *file = TFile::Open("./FEDMonitorHistos_evt1_spy121834.root");
  TFile *file = TFile::Open("./SpyDisplay.root");

  unsigned int runNumber = 128646;

  const unsigned int nEvts = 5;
  unsigned int evtNumber[nEvts] = {1,2,3,4,5};

  const unsigned int nDets = 11;
  unsigned int detIdVec[nDets] = {436265548,436265552,436265556,436265560,436265576, 
				  436265580,436265584,436265588,436265592,
				  369121566,369124733};

  int pair = -1; // -1 = all

  TString lBaseDir = "SiStripSpyDisplay/Display/run";
  lBaseDir += runNumber;
  lBaseDir += "_event";

  TLatex lat;
  gStyle->SetOptStat(0);
      
  TCanvas *myc = new TCanvas("myc","",800,400);


  TString lEvtDir;

  for (unsigned int iEv(0); iEv<nEvts; iEv++){

    lEvtDir = lBaseDir;
    lEvtDir += evtNumber[iEv];
    lEvtDir += "/detID_";

    TString lDetDir;

    for (unsigned int iDet(0); iDet<nDets; iDet++){

      unsigned int detId = detIdVec[iDet];
      lDetDir = lEvtDir;
      lDetDir += detId;
      lDetDir += "/detID_";
      lDetDir += detId;
      lDetDir += "_APVpair_";
      
      TString lDir;
      
      myc->cd();
      
      TH1F *p_scopeMode[3];
    
      for (unsigned int lPair(0); lPair<3; lPair++){
	if (pair >= 0 && lPair != pair) continue;
	lDir = lDetDir;
	lDir += lPair;
	lDir += "/";

	if (file->cd(lDir)){
	  p_scopeMode[lPair] = (TH1F*)gDirectory->Get("ScopeMode");
	  if (!p_scopeMode[lPair]) continue;
	  
	  p_scopeMode[lPair]->SetMaximum(1100);
	  p_scopeMode[lPair]->SetMinimum(0);
	  
	  TString leg = "Pair ";
	  leg += lPair;
	  
	  if (lPair == 0 || pair >= 0) {
	    p_scopeMode[lPair]->Draw();
	    TString lLabel = "DetID ";
	    lLabel += detId;
	    if (pair >= 0) {
	      lLabel += ", pair ";
	      lLabel += pair;
	    }
	    lat.DrawLatex(50,1000,lLabel);
	    if (!(pair >= 0)) lat.DrawLatex(200,900,leg);
	  }
	  else {
	    p_scopeMode[lPair]->SetLineColor(lPair+1);
	    p_scopeMode[lPair]->Draw("same");
	    lat.SetTextColor(lPair+1);
	    lat.DrawLatex(200,900-70*lPair,leg);
	  }
	  
	  file->cd();
	}
	else {
	  std::cout << " -- Pair " << lPair << " not found." << std::endl;
	}
	
      }//loop on pairs

      TString save = "PLOTS/";
      std::ostringstream lmkdir;
      lmkdir << "PLOTS/" << runNumber;
      mkdir(lmkdir.str().c_str(),0755);
      save += runNumber;
      save += "/Module";
      save += detId;
      save += "_evt";
      save += evtNumber[iEv];
      if (pair >= 0) {
	save += "_pair";
	save += pair;
      }
      save += ".png";

      myc->Print(save);

    }//loop on detids

  }//loop on evts

}//main
