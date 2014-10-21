#include "TrackPlots.h"
#include <vector>
#include <string>
#include <map>
#include "TPad.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TF1.h"
#include "TProfile2D.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TStyle.h"

void TrackPlots(const char* fullname,const char* module, const char* label, const char* postfix, const char* shortname, const char* outtrunk) {

  char modfull[300];
  sprintf(modfull,"%s%s",module,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);


  //  char fullname[300];
  //  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);

  TFile ff(fullname);

  // Colliding events

  
  CommonAnalyzer castat(&ff,"",modfull);

      TH1F* ntrk  = (TH1F*)castat.getObject("ntrk");
      if (ntrk) {
	ntrk->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/ntrk_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ntrk;
      }
      TH1F* pt  = (TH1F*)castat.getObject("pt");
      if (pt) {
	pt->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/pt_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->SetLogy(1);
	gPad->Print(plotfilename.c_str());
	gPad->SetLogy(0);
	delete pt;
      }
      gStyle->SetOptStat(11);
      TProfile2D* ptphieta  = (TProfile2D*)castat.getObject("ptphivseta");
      if (ptphieta) {
	ptphieta->Draw("colz");
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/ptphieta_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ptphieta;
      }
      gStyle->SetOptStat(1111);
      TH1F* phi  = (TH1F*)castat.getObject("phi");
      if (phi) {
	phi->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/phi_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete phi;
      }
      TH1F* eta  = (TH1F*)castat.getObject("eta");
      if (eta) {
	eta->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/eta_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete eta;
      }
      gStyle->SetOptStat(11);
      TH2F* phieta  = (TH2F*)castat.getObject("phivseta");
      if (phieta) {
	phieta->Draw("colz");
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/phieta_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete phieta;
      gStyle->SetOptStat(1111);
      }
      TH1F* nrhits  = (TH1F*)castat.getObject("nrhits");
      if (nrhits) {
	nrhits->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nrhits_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nrhits;
      }
      gStyle->SetOptStat(11);
      TProfile2D* nhitphieta  = (TProfile2D*)castat.getObject("nhitphivseta");
      if (nhitphieta) {
	nhitphieta->Draw("colz");
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nhitphieta_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nhitphieta;
      }
      gStyle->SetOptStat(1111);
      TH1F* nlosthits  = (TH1F*)castat.getObject("nlosthits");
      if (nlosthits) {
	nlosthits->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nlosthits_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nlosthits;
      }
      TH1F* npixelhits  = (TH1F*)castat.getObject("npixelhits");
      if (npixelhits) {
	npixelhits->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/npixelhits_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete npixelhits;
      }
      TH1F* nstriphits  = (TH1F*)castat.getObject("nstriphits");
      if (nstriphits) {
	nstriphits->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nstriphits_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nstriphits;
      }
      TH1F* nrlayers  = (TH1F*)castat.getObject("nlayers");
      if (nrlayers) {
	nrlayers->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nrlayers_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nrlayers;
      }
      gStyle->SetOptStat(11);
      TProfile2D* nlayerphieta  = (TProfile2D*)castat.getObject("nlayerphivseta");
      if (nlayerphieta) {
	nlayerphieta->Draw("colz");
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nlayerphieta_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nlayerphieta;
      }
      gStyle->SetOptStat(1111);
      TH1F* nlostlayers  = (TH1F*)castat.getObject("nlostlayers");
      if (nlostlayers) {
	nlostlayers->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nlostlayers_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nlostlayers;
      }
      TH1F* npixellayers  = (TH1F*)castat.getObject("npixellayers");
      if (npixellayers) {
	npixellayers->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/npixellayers_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete npixellayers;
      }
      TH1F* nstriplayers  = (TH1F*)castat.getObject("nstriplayers");
      if (nstriplayers) {
	nstriplayers->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/nstriplayers_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nstriplayers;
      }
      TH1F* hhpfrac  = (TH1F*)castat.getObject("hhpfrac");
      if (hhpfrac) {
	hhpfrac->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/hhpfrac_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->SetLogy(1);
	gPad->Print(plotfilename.c_str());
	gPad->SetLogy(0);
	delete hhpfrac;
      }
      TH1F* halgo  = (TH1F*)castat.getObject("algo");
      if (halgo) {
	halgo->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/halgo_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete halgo;
      }
      gStyle->SetOptStat(111);
      gStyle->SetOptFit(111);
      TProfile* hntrkvslumi  = (TProfile*)castat.getObject("ntrkvslumi");
      if (hntrkvslumi && hntrkvslumi->GetEntries()>0) {
	//	hntrkvslumi->Draw();
	hntrkvslumi->Fit("pol2","","",0.5,3.0);
	if(hntrkvslumi->GetFunction("pol2")) {
	  hntrkvslumi->GetFunction("pol2")->SetLineColor(kBlack);
	  hntrkvslumi->GetFunction("pol2")->SetLineWidth(1);
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/hntrkvslumi_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
      }
      gStyle->SetOptStat(1111);

      TH2D* hntrkvslumi2D  = (TH2D*)castat.getObject("ntrkvslumi2D");
      if (hntrkvslumi2D && hntrkvslumi2D->GetEntries()>0) {
	hntrkvslumi2D->Draw("colz");
	if(hntrkvslumi) {
	  hntrkvslumi->SetMarkerStyle(20);
	  hntrkvslumi->SetMarkerSize(.3);
	  hntrkvslumi->Draw("same");
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += shortname;
	plotfilename += "/hntrkvslumi2D_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += shortname;
	plotfilename += ".gif";
	gPad->SetLogz(1);
	gPad->Print(plotfilename.c_str());
	gPad->SetLogz(0);
	delete hntrkvslumi2D;
      }
      delete hntrkvslumi;
}
