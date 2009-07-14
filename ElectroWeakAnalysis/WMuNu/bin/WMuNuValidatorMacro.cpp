#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
//#include "TApplication.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPaveLabel.h"
#include "TPad.h"
#include "TH1D.h"
#include "TLegend.h"

int main(int argc, char** argv){

  if (argc!=3){
    printf("Usage: WMuNuValidatorMacro 'root_file_to_validate' 'reference_root_file'\n\n");
    printf("Example: WMuNuValidatorMacro WMuNu_histograms_31X.root WMuNu_histograms_22X.root\n");
    printf("Output: canvases in 'WMuNuValidation_*.root'\n");
    printf("        gif files in 'WMuNuValidation_*.gif'\n\n");
    return 1;
  }

  //TApplication* app = new TApplication("CMS Root Application", 0, 0);

  TString chfile = argv[1];
  TString chfileref = argv[2];
  TString cmssw_version = gSystem->Getenv("CMSSW_VERSION");
  TString chtitle = "WMuNu validation for " + cmssw_version;

  //TCanvas* c1 = new TCanvas("c1",chtitle.Data());
  TCanvas* c1 = new TCanvas("c1",chtitle.Data(),0,0,1024,768);

  TPaveLabel* paveTitle = new TPaveLabel(0.1,0.93,0.9,0.99, chtitle.Data());
  paveTitle->Draw();

  gStyle->SetOptTitle(1);
  gStyle->SetOptLogx(0);
  gStyle->SetOptLogy(1);
  gStyle->SetPadGridX(true);
  gStyle->SetPadGridY(true);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadBottomMargin(0.12);

  gStyle->SetOptFit(0000);
  gStyle->SetOptStat(1111111);

  TPad* pad[4];
  pad[0] = new TPad("pad_tl","The top-left pad",0.01,0.48,0.49,0.92); 
  pad[1] = new TPad("pad_tr","The top-right pad",0.51,0.48,0.99,0.92); 
  pad[2] = new TPad("pad_bl","The bottom-left pad",0.01,0.01,0.49,0.46); 
  pad[3] = new TPad("pad_br","The bottom-right pad",0.51,0.01,0.99,0.46); 
  pad[0]->Draw(); pad[1]->Draw(); pad[2]->Draw(); pad[3]->Draw();
                                                                                
  TLegend* leg = new TLegend(0.5,0.9,0.7,1.0);

  TFile* input_file = new TFile(chfile.Data(),"READONLY");
  TFile* input_fileref = new TFile(chfileref.Data(),"READONLY");
  bool first_plots_done = false;

  TDirectory* dir_before = input_file->GetDirectory("wmnSelFilter/BeforeCuts");
  TDirectory* dirref_before = input_fileref->GetDirectory("wmnSelFilter/BeforeCuts");
  TList* list_before = dir_before->GetListOfKeys();
  //list_before->Print();

  unsigned int list_before_size = list_before->GetSize();
  TString auxTitle = chtitle + ": BEFORE CUTS";
  for (unsigned int i=0; i<list_before_size; i+=4) {
      if (first_plots_done==true) c1->DrawClone();
      paveTitle->SetLabel(auxTitle.Data());
      for (unsigned int j=0; j<4; ++j) {
            if ((i+j)>=list_before_size) break;
            pad[j]->cd(); 

            TH1D* h1 = (TH1D*)dir_before->Get(list_before->At(i+j)->GetName()); 
            h1->SetLineColor(kBlue);
            h1->SetLineStyle(1);
            h1->SetLineWidth(3);
            h1->SetTitleSize(0.05,"X");
            h1->SetTitleSize(0.05,"Y");
            h1->SetXTitle(h1->GetTitle()); 
            h1->SetYTitle("");
            h1->SetTitle(""); 
            h1->SetTitleOffset(0.85,"X");

            TH1D* hr = (TH1D*)dirref_before->Get(list_before->At(i+j)->GetName()); 
            hr->SetLineColor(kRed);
            hr->SetLineStyle(2);
            hr->SetLineWidth(3);

            h1->Draw("hist");
            hr->DrawNormalized("samehist",h1->GetEntries());
            leg->Clear();
            leg->AddEntry(h1,cmssw_version.Data(),"L");
            leg->AddEntry(hr,"Reference","L");
            leg->Draw();
      }
      first_plots_done = true;
      c1->Modified();
      c1->Update();
      char chplot[80];
      sprintf(chplot,"WMuNuValidation_%s_BEFORECUTS_%d.root",cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
      sprintf(chplot,"WMuNuValidation_%s_BEFORECUTS_%d.gif",cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
  }

  TDirectory* dir_lastcut = input_file->GetDirectory("wmnSelFilter/LastCut");
  TDirectory* dirref_lastcut = input_fileref->GetDirectory("wmnSelFilter/LastCut");
  TList* list_lastcut = dir_lastcut->GetListOfKeys();
  //list_lastcut->Print();

  unsigned int list_lastcut_size = list_lastcut->GetSize();
  auxTitle = chtitle + ": AFTER N-1 CUTS";
  for (unsigned int i=0; i<list_lastcut_size; i+=4) {
      if (first_plots_done==true) c1->DrawClone();
      paveTitle->SetLabel(auxTitle.Data());
      for (unsigned int j=0; j<4; ++j) {
            if ((i+j)>=list_lastcut_size) break;
            pad[j]->cd(); 

            TH1D* h1 = (TH1D*)dir_lastcut->Get(list_lastcut->At(i+j)->GetName()); 
            h1->SetLineColor(kBlue);
            h1->SetLineWidth(3);
            h1->SetTitleSize(0.05,"X");
            h1->SetTitleSize(0.05,"Y");
            h1->SetXTitle(h1->GetTitle()); 
            h1->SetYTitle("");
            h1->SetTitle(""); 
            h1->SetTitleOffset(0.85,"X");

            TH1D* hr = (TH1D*)dirref_lastcut->Get(list_lastcut->At(i+j)->GetName()); 
            hr->SetLineColor(kRed);
            hr->SetLineStyle(2);
            hr->SetLineWidth(3);

            h1->Draw("hist");
            hr->DrawNormalized("samehist",h1->GetEntries());
            leg->Clear();
            leg->AddEntry(h1,cmssw_version.Data(),"L");
            leg->AddEntry(hr,"Reference","L");
            leg->Draw();
      }
      first_plots_done = true;
      c1->Modified();
      c1->Update();
      char chplot[80];
      sprintf(chplot,"WMuNuValidation_%s_LASTCUT_%d.root",cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
      sprintf(chplot,"WMuNuValidation_%s_LASTCUT_%d.gif",cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
  }

  //app->Run();
}


