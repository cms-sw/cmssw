#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"

#define NOISEPREFIX "Profile_NoiseFromCondDB__det__"
#define PEDESTALPREFIX "Profile_PedestalFromCondDB__det__"
#define OCCUPANCYPREFIX "ClusterDigiPosition__det__"

void printPlot(TH1D* hist, char* prefix, char* postfix);

int main(int argc, char *argv[]) {

  char* rootfilename;
  char* modulelistname;
  int pnbits;
  char* prefix;
  char* postfix;

  if(argc==6) {
    rootfilename = argv[1];
    modulelistname = argv[2];
    pnbits = atoi(argv[3]);
    prefix = argv[4];
    postfix = argv[5];
  }
  else {
    std::cout << "Wrong number of parameters " << argc << std::endl;
    return 1;
  }

  std::cout << "ready to go " << rootfilename << ' ' << modulelistname << std::endl;

  TFile* rootfile = new TFile(rootfilename,"READ");
  if(!rootfile) {
    std::cout << "Problems with input root file" << std::endl;
    return 2;
  }
  int detid;
  std::ifstream modulelist(modulelistname);

  std::stringstream outrootfilename;
  outrootfilename << prefix << "SummaryFile" << postfix << ".root";
  TFile* outrootfile = new TFile(outrootfilename.str().c_str(),"CREATE");

  while (modulelist >> detid) {
    std::cout << " ready to go with detid " << detid << " " << pnbits << std::endl;
    // bit 0: noise
    if(pnbits & 1) {
      std::stringstream histoname;
      histoname << NOISEPREFIX << detid;
      std::cout << " ready to go with histogram " << histoname.str() << std::endl;
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	std:: cout << histoname.str() << " found!" << std::endl;
	printPlot(hist,prefix,postfix);
	hist->Write();
      } else {  
	std:: cout << histoname.str() << " NOT found..." << std::endl;
      }
    }
    // bit 1: pedestal
    if(pnbits & 2) {
      std::stringstream histoname;
      histoname << PEDESTALPREFIX << detid;
      std::cout << " ready to go with histogram " << histoname.str() << std::endl;
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	std:: cout << histoname.str() << " found!" << std::endl;
	printPlot(hist,prefix,postfix);
	hist->Write();
      } else {  
	std:: cout << histoname.str() << " NOT found..." << std::endl;
      }
    }
    // bit 2: pedestal
    if(pnbits & 4) {
      std::stringstream histoname;
      histoname << OCCUPANCYPREFIX << detid;
      std::cout << " ready to go with histogram " << histoname.str() << std::endl;
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	std:: cout << histoname.str() << " found!" << std::endl;
	printPlot(hist,prefix,postfix);
	hist->Write();
      } else {  
	std:: cout << histoname.str() << " NOT found..." << std::endl;
      }
    }
  }

  outrootfile->Close();

  return 0;

}

void printPlot(TH1D* hist, char* prefix, char* postfix) {

  TCanvas* cc= new TCanvas;
  hist->Draw();
  std::stringstream filename;
  filename << prefix << hist->GetName() << postfix << ".png";
  cc->Print(filename.str().c_str());
  delete cc;

}
