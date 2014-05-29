#include "SeedMultiplicityPlots.h"
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <iostream>
#include "TPad.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TLegend.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"

void SeedMultiplicityComparisonPlot() {

  //  gStyle->SetOptStat(111111);
  TFile f1("/afs/cern.ch/cms/tracking/output/rootfiles/seedmultiplicity_Run2011A_express_ge1711_v3_v2.root");
  TFile f2("/afs/cern.ch/cms/tracking/output/rootfiles/seedmultiplicity_highmult_default_Run2011A_express_ge1711_v3_v2.root");
  TFile f3("/afs/cern.ch/cms/tracking/output/rootfiles/seedmultiplicity_highmult_Run2011A_express_ge1711_v3_v2.root");

  CommonAnalyzer castat1(&f1,"","seedmultiplicitymonitor");
  CommonAnalyzer castat2(&f2,"","seedmultiplicitymonitor");
  CommonAnalyzer castat3(&f3,"","seedmultiplicitymonitor");

  TH2F* iter3Pixel1 = (TH2F*)castat1.getObject("thTripletsPixel");
  TH2F* iter3Pixel2 = (TH2F*)castat2.getObject("thTripletsPixel");
  TH2F* iter3Pixel3 = (TH2F*)castat3.getObject("thTripletsPixel");
  
  if(iter3Pixel1 && iter3Pixel2 && iter3Pixel3) {
    TProfile* iter3Pixel1_prof = iter3Pixel1->ProfileX("prof1");
    TProfile* iter3Pixel2_prof = iter3Pixel2->ProfileX("prof2");
    TProfile* iter3Pixel3_prof = iter3Pixel3->ProfileX("prof3");

    iter3Pixel1_prof->SetMarkerColor(kBlack);    iter3Pixel1_prof->SetLineColor(kBlack);
    iter3Pixel2_prof->SetMarkerColor(kBlue);    iter3Pixel2_prof->SetLineColor(kBlue);
    iter3Pixel3_prof->SetMarkerColor(kRed);    iter3Pixel3_prof->SetLineColor(kRed);

    new TCanvas("profile","profile");

    iter3Pixel1_prof->GetXaxis()->SetRangeUser(0,10000);
    iter3Pixel1_prof->GetXaxis()->SetTitle("clusters");    iter3Pixel1_prof->GetYaxis()->SetTitle("seeds");
    
    

    TLegend* leg = new TLegend(.2,.6,.5,.85,"Seeds vs clusters");
    leg->AddEntry(iter3Pixel1_prof->DrawCopy(),"standard RECO","l");
    leg->AddEntry(iter3Pixel2_prof->DrawCopy("same"),"standard RECO (high stat)","l");
    leg->AddEntry(iter3Pixel3_prof->DrawCopy("same"),"iter2 thr = 2M","l");
    leg->Draw();

    /*
    std::cout << iter3Pixel1->GetTitle() << std::endl;
    std::cout << iter3Pixel1->GetEntries() << std::endl;
    iter3Pixel1->DrawCopy();
    iter3Pixel1->GetXaxis()->SetRangeUser(0,10000);
    iter3Pixel2->DrawCopy("same");
    iter3Pixel3->DrawCopy("same");
    */

  }


  TH1F* iter3_2 = (TH1F*)castat2.getObject("thTriplets");
  TH1F* iter3_3 = (TH1F*)castat3.getObject("thTriplets");

  if(iter3_2 && iter3_3) {

    iter3_2->SetLineColor(kBlue);
    iter3_3->SetLineColor(kRed);
    
    new TCanvas("iter3","iter3");
    
    gPad->SetLogy(1);
    TLegend* legiter3 = new TLegend(.5,.6,.85,.85,"iter3 seeds");
    legiter3->AddEntry(iter3_2->DrawCopy(),"standard RECO","l");
    legiter3->AddEntry(iter3_3->DrawCopy("same"),"iter2 thr = 2M","l");
    legiter3->Draw();

  }

  TH1F* iter2_2 = (TH1F*)castat2.getObject("secTriplets");
  TH1F* iter2_3 = (TH1F*)castat3.getObject("secTriplets");

  if(iter2_2 && iter2_3) {

    iter2_2->SetLineColor(kBlue);
    iter2_3->SetLineColor(kRed);
    
    new TCanvas("iter2","iter2");
    
    gPad->SetLogy(1);
    TLegend* legiter2 = new TLegend(.5,.6,.85,.85,"iter2 seeds");
    legiter2->AddEntry(iter2_2->DrawCopy(),"standard RECO","l");
    legiter2->AddEntry(iter2_3->DrawCopy("same"),"iter2 thr = 2M","l");
    legiter2->Draw();

  }
  //  gStyle->SetOptStat(1111);
}

void SeedMultiplicityPlots(const char* fullname,const char* module, const char* postfix, const char* shortname, const char* outtrunk) {

  std::cout << shortname << module << postfix <<  std::endl;

  char modfull[300];
  sprintf(modfull,"%s%s",module,postfix);

  char dirname[400];
  //  sprintf(dirname,"%s%s",family,filename);
  sprintf(dirname,"%s%s","seedmultiplicity_",shortname);

  //  char fullname[300];
  //  if(strlen(family)==0) {  sprintf(fullname,"rootfiles/seedmultiplicity_%s.root",filename);}
  //  else {  sprintf(fullname,"rootfiles/%s.root",dirname); }



  //  std::string workdir = outtrunk ;
  //  workdir += dirname;
  //  gSystem->cd(workdir.c_str());
  //  gSystem->MakeDirectory(labfull);
  //  gSystem->cd("/afs/cern.ch/cms/tracking/output");


  TFile ff(fullname);

  // Colliding events

  gStyle->SetOptStat(111111);
  
  CommonAnalyzer castat(&ff,"",modfull);

  std::cout << "ready" << std::endl;
    
  TH1F* iter0 = (TH1F*)castat.getObject("newSeedFromTriplets");
  if(iter0) {
	iter0->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter0_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter0;
	gPad->SetLogy(0);
  }

  TH1F* iter1 = (TH1F*)castat.getObject("newSeedFromPairs");
  if(iter1) {
	iter1->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter1_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter1;
	gPad->SetLogy(0);
  }

  TH1F* iter2 = (TH1F*)castat.getObject("secTriplets");
  if(iter2) {
	iter2->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter2_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter2;
	gPad->SetLogy(0);
  }

  TH1F* iter3 = (TH1F*)castat.getObject("thTriplets");
  if(iter3) {
	iter3->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3;
	gPad->SetLogy(0);
  }

  TH1F* iter3A = (TH1F*)castat.getObject("thTripletsA");
  if(iter3A) {
	iter3A->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3A_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3A;
	gPad->SetLogy(0);
  }

  TH1F* iter3B = (TH1F*)castat.getObject("thTripletsB");
  if(iter3B) {
	iter3B->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3B_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3B;
	gPad->SetLogy(0);
  }

  TH1F* iter4 = (TH1F*)castat.getObject("fourthPLSeeds");
  if(iter4) {
	iter4->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter4_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter4;
	gPad->SetLogy(0);
  }

  TH1F* iter5 = (TH1F*)castat.getObject("fifthSeeds");
  if(iter5) {
	iter5->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter5_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter5;
	gPad->SetLogy(0);
  }

  TH2F* iter0TK = (TH2F*)castat.getObject("newSeedFromTripletsTK");
  if(iter0TK) {
	iter0TK->Draw("colz");
	iter0TK->GetYaxis()->SetRangeUser(0,1000);   iter0TK->GetYaxis()->SetTitle("seeds");
	iter0TK->GetXaxis()->SetRangeUser(0,50000); iter0TK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter0TK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter0TK;
	gPad->SetLogz(0);
  }

  TH2F* iter1TK = (TH2F*)castat.getObject("newSeedFromPairsTK");
  if(iter1TK) {
	iter1TK->Draw("colz");
	iter1TK->GetYaxis()->SetTitle("seeds");
	iter1TK->GetXaxis()->SetRangeUser(0,50000); iter1TK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter1TK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter1TK;
	gPad->SetLogz(0);
  }

  TH2F* iter2TK = (TH2F*)castat.getObject("secTripletsTK");
  if(iter2TK) {
	iter2TK->Draw("colz");
	iter2TK->GetYaxis()->SetRangeUser(0,50000);  iter2TK->GetYaxis()->SetTitle("seeds");
	iter2TK->GetXaxis()->SetRangeUser(0,50000); iter2TK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter2TK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter2TK;
	gPad->SetLogz(0);
  }

  TH2F* iter3TK = (TH2F*)castat.getObject("thTripletsTK");
  if(iter3TK) {
	iter3TK->Draw("colz");
	//	iter3TK->GetYaxis()->SetRangeUser(0,30000);  
	iter3TK->GetYaxis()->SetTitle("seeds");
	iter3TK->GetXaxis()->SetRangeUser(0,50000); iter3TK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3TK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3TK;
	gPad->SetLogz(0);
  }

  TH2F* iter3ATK = (TH2F*)castat.getObject("thTripletsATK");
  if(iter3ATK) {
	iter3ATK->Draw("colz");
	//	iter3ATK->GetYaxis()->SetRangeUser(0,30000);  
	iter3ATK->GetYaxis()->SetTitle("seeds");
	iter3ATK->GetXaxis()->SetRangeUser(0,50000); iter3ATK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3ATK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3ATK;
	gPad->SetLogz(0);
  }

  TH2F* iter3BTK = (TH2F*)castat.getObject("thTripletsBTK");
  if(iter3BTK) {
	iter3BTK->Draw("colz");
	//	iter3BTK->GetYaxis()->SetRangeUser(0,30000);  
	iter3BTK->GetYaxis()->SetTitle("seeds");
	iter3BTK->GetXaxis()->SetRangeUser(0,50000);  iter3BTK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3BTK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3BTK;
	gPad->SetLogz(0);
  }

  TH2F* iter4TK = (TH2F*)castat.getObject("fourthPLSeedsTK");
  if(iter4TK) {
	iter4TK->Draw("colz");
	iter4TK->GetYaxis()->SetTitle("seeds");
	iter4TK->GetXaxis()->SetRangeUser(0,50000); iter4TK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter4TK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter4TK;
	gPad->SetLogz(0);
  }

  TH2F* iter5TK = (TH2F*)castat.getObject("fifthSeedsTK");
  if(iter5TK) {
	iter5TK->Draw("colz");
	iter5TK->GetYaxis()->SetRangeUser(0,30000); iter5TK->GetYaxis()->SetTitle("seeds");
	iter5TK->GetXaxis()->SetRangeUser(0,50000); iter5TK->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter5TK_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter5TK;
	gPad->SetLogz(0);
  }

  TH2F* iter0Pixel = (TH2F*)castat.getObject("newSeedFromTripletsPixel");
  if(iter0Pixel) {
	iter0Pixel->Draw("colz");
	iter0Pixel->GetYaxis()->SetRangeUser(0,1000);   iter0Pixel->GetYaxis()->SetTitle("seeds");
	iter0Pixel->GetXaxis()->SetRangeUser(0,10000); iter0Pixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter0Pixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter0Pixel;
	gPad->SetLogz(0);
  }

  TH2F* iter1Pixel = (TH2F*)castat.getObject("newSeedFromPairsPixel");
  if(iter1Pixel) {
	iter1Pixel->Draw("colz");
	iter1Pixel->GetYaxis()->SetTitle("seeds");
	iter1Pixel->GetXaxis()->SetRangeUser(0,10000); iter1Pixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter1Pixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter1Pixel;
	gPad->SetLogz(0);
  }

  TH2F* iter2Pixel = (TH2F*)castat.getObject("secTripletsPixel");
  if(iter2Pixel) {
	iter2Pixel->Draw("colz");
	iter2Pixel->GetYaxis()->SetRangeUser(0,50000);  iter2Pixel->GetYaxis()->SetTitle("seeds");
	iter2Pixel->GetXaxis()->SetRangeUser(0,10000); iter2Pixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter2Pixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter2Pixel;
	gPad->SetLogz(0);
  }

  TH2F* iter3Pixel = (TH2F*)castat.getObject("thTripletsPixel");
  if(iter3Pixel) {
	iter3Pixel->Draw("colz");
	//	iter3Pixel->GetYaxis()->SetRangeUser(0,30000);  
	iter3Pixel->GetYaxis()->SetTitle("seeds");
	iter3Pixel->GetXaxis()->SetRangeUser(0,10000); iter3Pixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3Pixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3Pixel;
	gPad->SetLogz(0);
  }

  TH2F* iter3APixel = (TH2F*)castat.getObject("thTripletsAPixel");
  if(iter3APixel) {
	iter3APixel->Draw("colz");
	//	iter3APixel->GetYaxis()->SetRangeUser(0,30000);  
	iter3APixel->GetYaxis()->SetTitle("seeds");
	iter3APixel->GetXaxis()->SetRangeUser(0,10000); iter3APixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3APixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3APixel;
	gPad->SetLogz(0);
  }

  TH2F* iter3BPixel = (TH2F*)castat.getObject("thTripletsBPixel");
  if(iter3BPixel) {
	iter3BPixel->Draw("colz");
	//	iter3BPixel->GetYaxis()->SetRangeUser(0,30000);  
	iter3BPixel->GetYaxis()->SetTitle("seeds");
	iter3BPixel->GetXaxis()->SetRangeUser(0,10000);  iter3BPixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter3BPixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter3BPixel;
	gPad->SetLogz(0);
  }

  TH2F* iter4Pixel = (TH2F*)castat.getObject("fourthPLSeedsPixel");
  if(iter4Pixel) {
	iter4Pixel->Draw("colz");
	iter4Pixel->GetYaxis()->SetTitle("seeds");
	iter4Pixel->GetXaxis()->SetRangeUser(0,10000); iter4Pixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter4Pixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter4Pixel;
	gPad->SetLogz(0);
  }

  TH2F* iter5Pixel = (TH2F*)castat.getObject("fifthSeedsPixel");
  if(iter5Pixel) {
	iter5Pixel->Draw("colz");
	iter5Pixel->GetYaxis()->SetRangeUser(0,30000); iter5Pixel->GetYaxis()->SetTitle("seeds");
	iter5Pixel->GetXaxis()->SetRangeUser(0,10000); iter5Pixel->GetXaxis()->SetTitle("clusters");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/iter5Pixel_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete iter5Pixel;
	gPad->SetLogz(0);
  }
  gStyle->SetOptStat(1111);

}

