#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TRint.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPaveLabel.h"
#include "TPad.h"
#include "TH1D.h"
#include "TLegend.h"

int printUsage(){
    printf("Usage: WMuNuValidatorMacro [-lbh] 'root_file_to_validate' 'reference_root_file'\n\n");

    printf("\tOptions:\t -l ==> linear scale for Y axes (default is log-scale)\n");
    printf("\t        \t -b ==> run in batch (no graphics)\n");
    printf("\t        \t -h ==> print this message\n\n");

    printf("\tInput files:\t Created via '*Validator.py' configuration files in:\n");
    printf("\t            \t   $CMSSW_BASE/src/ElectroWeakAnalysis/WMuNu/test/\n\n");

    printf("\tOutput: \t Canvases: './WMuNuValidation_$CMSSW_VERSION_*.root'\n");
    printf("\t        \t Gifs:     './WMuNuValidation_$CMSSW_VERSION_*.gif'\n\n");

    return 1;
}

int main(int argc, char** argv){

  TString chfile;
  TString chfileref;

  int ntrueargs = 0;
  bool logyFlag = true;
  for (int i=1; i<argc; ++i) {
      if (argv[i][0] == '-') {
            if (argv[i][1]=='l') logyFlag = false;
            else if (argv[i][1]=='b') gROOT->SetBatch();
            else if (argv[i][1]=='h') return printUsage();
      } else {
            ntrueargs += 1;
            if (ntrueargs==1) chfile = argv[i];
            else if (ntrueargs==2) chfileref = argv[i];
      }
  }

  if (ntrueargs!=2) return printUsage();

  TRint* app = new TRint("CMS Root Application", 0, 0);

  TString cmssw_version = gSystem->Getenv("CMSSW_VERSION");
  TString chsample = "WMuNu";
  TString chtitle = chsample + " validation for " + cmssw_version;

  //TCanvas* c1 = new TCanvas("c1",chtitle.Data());
  TCanvas* c1 = new TCanvas("c1",chtitle.Data(),0,0,1024,768);

  TPaveLabel* paveTitle = new TPaveLabel(0.1,0.93,0.9,0.99, chtitle.Data());
  paveTitle->Draw();

  gStyle->SetOptLogy(logyFlag);
  gStyle->SetPadGridX(true);
  gStyle->SetPadGridY(true);
  gStyle->SetOptStat(1111111);

  TPad* pad[4];
  pad[0] = new TPad("pad_tl","The top-left pad",0.01,0.48,0.49,0.92); 
  pad[1] = new TPad("pad_tr","The top-right pad",0.51,0.48,0.99,0.92); 
  pad[2] = new TPad("pad_bl","The bottom-left pad",0.01,0.01,0.49,0.46); 
  pad[3] = new TPad("pad_br","The bottom-right pad",0.51,0.01,0.99,0.46); 
  for (unsigned int i=0; i<4; ++i) pad[i]->Draw();
                                                                                
  TLegend* leg = new TLegend(0.5,0.9,0.7,1.0);

  TFile* input_file = new TFile(chfile.Data(),"READONLY");
  TFile* input_fileref = new TFile(chfileref.Data(),"READONLY");
  bool first_plots_done = false;

  TDirectory* dir_before = input_file->GetDirectory("wmnSelFilter/BeforeCuts");
  TDirectory* dirref_before = input_fileref->GetDirectory("wmnSelFilter/BeforeCuts");
  TList* list_before = dir_before->GetListOfKeys();
  list_before->Print();

  unsigned int list_before_size = list_before->GetSize();
  TString auxTitle = chtitle + ": BEFORE CUTS";
  for (unsigned int i=0; i<list_before_size; i+=4) {
      if (first_plots_done==true) c1->DrawClone();
      paveTitle->SetLabel(auxTitle.Data());
      for (unsigned int j=0; j<4; ++j) {
            pad[j]->cd(); 
            pad[j]->Clear(); 
            if ((i+j)>=list_before_size) continue;

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
            hr->DrawNormalized("samehist",h1->Integral());
            leg->Clear();
            leg->AddEntry(h1,cmssw_version.Data(),"L");
            leg->AddEntry(hr,"Reference","L");
            leg->Draw();
      }
      first_plots_done = true;
      c1->Modified();
      c1->Update();
      char chplot[80];
      sprintf(chplot,"%sValidation_%s_BEFORECUTS_%d.root",chsample.Data(),cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
      sprintf(chplot,"%sValidation_%s_BEFORECUTS_%d.gif",chsample.Data(),cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
  }

  TDirectory* dir_lastcut = input_file->GetDirectory("wmnSelFilter/LastCut");
  TDirectory* dirref_lastcut = input_fileref->GetDirectory("wmnSelFilter/LastCut");
  TList* list_lastcut = dir_lastcut->GetListOfKeys();
  list_lastcut->Print();

  unsigned int list_lastcut_size = list_lastcut->GetSize();
  auxTitle = chtitle + ": AFTER N-1 CUTS";
  for (unsigned int i=0; i<list_lastcut_size; i+=4) {
      if (first_plots_done==true) c1->DrawClone();
      paveTitle->SetLabel(auxTitle.Data());
      for (unsigned int j=0; j<4; ++j) {
            pad[j]->cd(); 
            pad[j]->Clear(); 
            if ((i+j)>=list_lastcut_size) continue;

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
            hr->DrawNormalized("samehist",h1->Integral());
            leg->Clear();
            leg->AddEntry(h1,cmssw_version.Data(),"L");
            leg->AddEntry(hr,"Reference","L");
            leg->Draw();
      }
      first_plots_done = true;
      c1->Modified();
      c1->Update();
      char chplot[80];
      sprintf(chplot,"%sValidation_%s_LASTCUT_%d.root",chsample.Data(),cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
      sprintf(chplot,"%sValidation_%s_LASTCUT_%d.gif",chsample.Data(),cmssw_version.Data(),i/4);
      c1->SaveAs(chplot);
  }

  if (!gROOT->IsBatch()) app->Run();

  return 0;
}
