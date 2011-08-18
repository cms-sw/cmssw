#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

#include "TH1F.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TString.h"
#include "TDirectory.h"

int main(int argc, char** argv) {//main

  if (argc < 3){
    std::cout << "Usage: " << argv[0] 
	      << " <file> <run number>"
	      << " <optional: local file>"
	      << std::endl;
    return 0;
  }


  unsigned int run;
  std::istringstream(argv[2])>>run;

  bool lLocal=0;
  if (argc>3) std::istringstream(argv[3])>>lLocal;

  std::ofstream outfile;
  TString outName = "FEDChannelErrors_run";
  outName += run;
  outName += ".txt";
  outfile.open(outName,std::ios::out);

  if (!outfile) {
    std::cerr << "Cannot open file " << outName << " for writting. Please debug! " << std::endl;
    return 0;
  }

  TString histNames[7] = {"OOSBits","UnlockedBits",
			  "APVAddressErrorBits","APVErrorBits",
			  "BadMajorityAddresses","FEMissing","FEOverflows"};

  TString fileName = argv[1];

  TFile *f = TFile::Open(fileName);

  if (!f) {
    std::cout << "Cannot open file " << fileName << " for reading ! Exiting..." << std::endl;
    return 0;
  }

  TString dirName = "DQMData/";
  if (!lLocal) {
    dirName += "Run ";
    dirName += run;
    dirName += "/";
  }
  dirName += "SiStrip/";
  if (!lLocal) dirName += "Run summary/";
  dirName += "ReadoutView/";
  std::cout << "Directory " << dirName << std::endl;

  if (!f->cd(dirName)) {
    std::cerr << "Folder not found, please modify source code " << __FILE__ << ", variable dirName!" << std::endl;
    return 0;
  }

  //looking for object with name chNames etc...
  TString normDir = dirName;
  normDir += "FedSummary/";
  if (!f->cd(normDir)) {
    std::cerr << "Folder not found, please modify source code " << __FILE__ << ", line " << __LINE__ << std::endl;
    return 0;
  };

  TH1F *hNorm = (TH1F*)gDirectory->Get("FED/nFEDErrors");
  double norm = hNorm->GetEntries(); 

  outfile << " - File contains " << norm << " events." << std::endl
	  << " - CHANNEL/FE ARE REPORTED IN INTERNAL NUMBERING SCHEME (0-95/0-7, FE #0 is channel 0-11)" << std::endl; 

  for (unsigned int ifed(50); ifed<500;ifed++){//loop on FEDs

    TString fedDir = dirName;
    fedDir += "FrontEndDriver";
    fedDir += ifed;

    if (!f->cd(fedDir)) continue;
    else {
      outfile << " - Errors detected for FED " << ifed << std::endl;
     }

    TDirectory *current = gDirectory;

    //channel histos
    for (unsigned int iHist(0); iHist<7; iHist++){//loop on histos
      TString objName = histNames[iHist];
      objName += "ForFED";
      objName += ifed;
      TH1F *obj = (TH1F*)current->Get(objName);

      if (!obj) {
	std::cout << "Warning, histogram " << objName << " not found..." << std::endl;
	continue;//return 0;
      }
      else {
	if (obj->GetEntries() != 0) {
	  for (int bin(1); bin<obj->GetNbinsX()+1; bin++){
	    if (obj->GetBinContent(bin)>0){ 
	      outfile << " --- FED ID " << ifed ;
	      if (iHist < 2) outfile << ", channel " << bin-1 ;
	      else if (iHist < 4) outfile << ", APV " << bin-1 
					  << " (channel " << static_cast<int>((bin-1)/2.) << ")";
	      else outfile << ", FE " << bin-1 ;
	      std::ostringstream message;
	      message << " in " << histNames[iHist] << " error for " ;
	      outfile << message.str() << obj->GetBinContent(bin)/norm*100. << "% of events." 
		      << std::endl;
	    }
	  }
	}
      }
    }//loop on histos

  }//loop on feds


  return 1;

}//main
