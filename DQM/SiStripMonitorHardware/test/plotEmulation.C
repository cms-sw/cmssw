#include "iostream"
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


void plotEmulation(){//main

  TFile *file = TFile::Open("./SpyDisplay.root");

  unsigned int runNumber = 128646;

  const unsigned int nEvts = 9;
  unsigned int evtNumber[nEvts] = {1,56,100,186,228,289,342,378,389};

  const unsigned int nDets = 11;
  unsigned int detId[nDets] = {436265548,436265552,436265556,436265560,436265576, 
			       436265580,436265584,436265588,436265592,
			       369121566,369124733};


  TString lBaseDir = "SiStripSpyDisplay/Display/run";
  lBaseDir += runNumber;
  lBaseDir += "_event";

  gStyle->SetOptStat(0);

  TLatex lat;
 
  TCanvas *myc = new TCanvas("myc","",800,800);
  myc->Divide(2,2);
 
  TString lEvtDir;

  for (unsigned int iDet(0); iDet<nDets; iDet++){

    TH1F *p_mode[4][nEvts];

    for (unsigned int iEv(0); iEv<nEvts; iEv++){

      lEvtDir = lBaseDir;
      lEvtDir += evtNumber[iEv];
      lEvtDir += "/detID_";

      TString lDetDir;

      lDetDir = lEvtDir;
      lDetDir += detId[iDet];
      lDetDir += "/";

      if (file->cd(lDetDir)){
	p_mode[0][iEv] = (TH1F*)gDirectory->Get("ReorderedModuleRaw");
	p_mode[1][iEv] = (TH1F*)gDirectory->Get("PedestalValues");
	p_mode[2][iEv] = (TH1F*)gDirectory->Get("PostPedestal");
	p_mode[3][iEv] = (TH1F*)gDirectory->Get("PostCommonMode");
      }
      else {
	std::cout << " -- Dir " << lDetDir << " not found." << std::endl;
      }

      for (unsigned int i(0); i<4; i++){
	myc->cd(i+1);
	if (!p_mode[i][iEv]) continue;
	p_mode[i][iEv]->SetLineColor(iEv+1);
	if (iEv == 0) p_mode[i][iEv]->Draw();
	else p_mode[i][iEv]->Draw("same");
      }

 
    }//loop on evts

    TString save = "PLOTS/";
    std::ostringstream lmkdir;
    lmkdir << "PLOTS/" << runNumber;
    mkdir(lmkdir.str().c_str(),0755);
    save += runNumber;
    save += "/FEDEmulation_";
    save += detId[iDet];
    save += ".png";
    
    myc->Print(save);
      


  }//loop on detids


}//main
