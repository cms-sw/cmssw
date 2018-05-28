#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "TFile.h"
#include "TH1D.h"
#include "TDirectoryFile.h"
#include "TCanvas.h"

#define NOISEPREFIX "Profile_NoiseFromCondDB__det__"
#define PEDESTALPREFIX "Profile_PedestalFromCondDB__det__"
#define OCCUPANCYPREFIX "ClusterDigiPosition__det__"

void printPlot(TH1D* hist, char* prefix, char* postfix);
int getChannelNumber(int ival);

int main(int argc, char *argv[]) {

  char* rootfilename;
  char* modulelistname;
  int pnbits;
  char* prefix;
  char* postfix;

  bool debugPrint = false;
  bool imagePrint = false;

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

  if (debugPrint) std::cout << "ready to go " << rootfilename << ' ' << modulelistname << std::endl;

  TFile* rootfile = TFile::Open(rootfilename,"READ");
  if(!rootfile) {
    std::cout << "Problems with input root file" << std::endl;
    return 2;
  }
  int detid;
  std::ifstream modulelist(modulelistname);

  std::stringstream outrootfilename;
  outrootfilename << prefix << "SummaryFile" << postfix << ".root";
  TFile* outrootfile = TFile::Open(outrootfilename.str().c_str(),"RECREATE");
  
  TH1D* th_summary = nullptr;
  Double_t TotalEvents = 0.0;
  Double_t TotalDigis = 0.0;

  TH1D* TotEvents= new TH1D("TotEvents","TotEvents",1,0,1);

  if(pnbits & 4) {
    TDirectoryFile* tdir = (TDirectoryFile*) rootfile->FindObjectAny("AlCaReco");
    TH1D* hist = (TH1D*)tdir->FindObjectAny("TotalNumberOfCluster__TIB");
    if (hist) {
      TotalEvents = hist->GetEntries();
      TotEvents->SetBinContent(1,TotalEvents);
      TotEvents->Write();
    }
  }    
  while (modulelist >> detid) {
    if (debugPrint) std::cout << " ready to go with detid " << detid << " " << pnbits << std::endl;
    // bit 0: noise
    if(pnbits & 1) {
      std::stringstream histoname;
      histoname << NOISEPREFIX << detid;
      if (debugPrint) std::cout << " ready to go with histogram " << histoname.str() << std::endl;
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	if (debugPrint) std::cout << histoname.str() << " found!" << std::endl;
	if (imagePrint) printPlot(hist,prefix,postfix);
	hist->Write();
      } else {  
	std::cout << histoname.str() << " NOT found..." << std::endl;
      }
    }
    // bit 1: pedestal
    if(pnbits & 2) {
      std::stringstream histoname;
      histoname << PEDESTALPREFIX << detid;
    if (debugPrint) std::cout << " ready to go with histogram " << histoname.str() << std::endl;
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	if (debugPrint) std::cout << histoname.str() << " found!" << std::endl;
        if (imagePrint) printPlot(hist,prefix,postfix);
	hist->Write();
      } else {  
	std::cout << histoname.str() << " NOT found..." << std::endl;
      }
    }
    // bit 2: Occupancy
    if(pnbits & 4) {
      std::stringstream histoname;
      histoname << OCCUPANCYPREFIX << detid;
      std::string SummaryName= "ClusterDigiPosition__det__Summary";

    if (debugPrint) std::cout << " ready to go with histogram " << histoname.str() << std::endl;
    if (th_summary == nullptr)  th_summary = new TH1D(SummaryName.c_str(), SummaryName.c_str(), 768, 0.5, 768.5);
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	if (debugPrint) std::cout << histoname.str() << " found!" <<  hist->GetEntries() << std::endl;
        for (int i = 1; i < hist->GetNbinsX()+1; i++) {
	  th_summary->AddBinContent(i, hist->GetBinContent(i));
          TotalDigis += hist->GetBinContent(i);
	}
	hist->SetLineColor(2);
	if (imagePrint) printPlot(hist,prefix,postfix);
	hist->Write();
      } else {  
	std::cout << histoname.str() << " NOT found..." << std::endl;
      }
    }
    // bit 3 : reorder
    if(pnbits & 8) {
      std::stringstream histoname;
      histoname << OCCUPANCYPREFIX << detid;
    if (debugPrint) std::cout << " ready to go with histogram " << histoname.str() << std::endl;
      TH1D* hist = (TH1D*)rootfile->FindObjectAny(histoname.str().c_str());
      if(hist) { 
	if (debugPrint) std::cout << histoname.str() << " found!" << std::endl;
	std::stringstream histoname_reorder;
        histoname_reorder << OCCUPANCYPREFIX <<"_reorder_"<< detid;
        TH1D* hist_reorder = new TH1D(histoname_reorder.str().c_str(), histoname_reorder.str().c_str(), 
				      hist->GetXaxis()->GetNbins(), hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
        for (int i = 0; i < hist_reorder->GetNbinsX(); i++) {
          int chan = getChannelNumber(i);
          hist_reorder->SetBinContent(i+1,hist->GetBinContent(chan));
	}
	hist->Write();
	hist_reorder->Write();
      } else {  
	std::cout << histoname.str() << " NOT found..." << std::endl;
      }
    }

  }
  if (th_summary) {
    std::string fname = rootfilename;
    std::string run = fname.substr(fname.find("R000")+4,6);
    if (TotalEvents) {
      Double_t fac = 1.0/TotalEvents;
      th_summary->Scale(fac);
      std::cout << " Run Number " << run << " Events " << TotalEvents << " Total # of Digis " << TotalDigis << " Av. Digis " << TotalDigis*fac << std::endl;
      th_summary->SetEntries(TotalDigis*fac);
	
    }
    th_summary->Write();
    printPlot(th_summary,prefix,postfix);
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
int getChannelNumber(int ival) {

  int chan=int(32*fmod(int(fmod(ival,256.)/2.),4.) +
	      8*int(int(fmod(ival,256.)/2.)/4.) -
	      31*int(int(fmod(ival,256.)/2.)/16.) +
	      fmod(fmod(ival,256.),2.)*128 +
	      int(ival/256)*256);
  return chan;
}
