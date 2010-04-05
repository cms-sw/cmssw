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
#include "TPaveStats.h"
int printUsage(){
    printf("Usage: EWKMuValidatorMacro [-lbh] 'root_file_to_validate' 'MinimumBias MC_root_file' 'directory_name'\n\n");

    printf("\tOptions:\t -l ==> linear scale for Y axes (default is log-scale)\n");
    printf("\t        \t -b ==> run in batch (no graphics)\n");
    printf("\t        \t -n ==> normalize MinimumBias MC to data (default = false)\n");
    printf("\t        \t -h ==> print this message\n\n");


    printf("\tInput files:\t Created via '*Validator.py' configuration files in:\n");
    printf("\t            \t   $CMSSW_BASE/src/ElectroWeakAnalysis/WMuNu/test/\n\n");

    printf("\tOutput: \t Canvases: './EWKMuValidation_$CMSSW_VERSION_*.root'\n");
    printf("\t        \t Gifs:     './EWKMuValidation_$CMSSW_VERSION_*.gif'\n\n");

    return 1;
}

int main(int argc, char** argv){

  TString chfile;
  TString chfileref;
  TString DirectoryLast;


  int ntrueargs = 0;
  bool logyFlag = true;
  bool normalize = false;

  for (int i=1; i<argc; ++i) {
      if (argv[i][0] == '-') {
            if (argv[i][1]=='l') logyFlag = false;
            else if (argv[i][1]=='b') gROOT->SetBatch();
            else if (argv[i][1]=='h') return printUsage();
            else if (argv[i][1]=='n') normalize=true;

      } else {
            ntrueargs += 1;
            if (ntrueargs==1) chfile = argv[i];
            else if (ntrueargs==2) chfileref = argv[i];
            else if (ntrueargs==3) DirectoryLast = argv[i];

      }
  }

  if (ntrueargs!=3) return printUsage();

  TRint* app = new TRint("CMS Root Application", 0, 0);

  TString cmssw_version = gSystem->Getenv("CMSSW_VERSION");
  TString chsample = "EWKMu";
  TString chtitle = chsample + " validation for " + cmssw_version;




  //TCanvas* c1 = new TCanvas("c1",chtitle.Data());
  TCanvas* c1 = new TCanvas("c1",chtitle.Data(),0,0,1024,768);
  c1->SetFillColor(0);
  TPaveLabel* paveTitle = new TPaveLabel(0.1,0.93,0.9,0.99, chtitle.Data());
  paveTitle->Draw();
  paveTitle->SetFillColor(0);
  gStyle->SetOptLogy(logyFlag);
  gStyle->SetPadGridX(true);
  gStyle->SetPadGridY(true);
  gStyle->SetOptStat(1111111);
//  gStyle->SetFillColor(0);

  TPad* pad[4];
  pad[0] = new TPad("pad_tl","The top-left pad",0.01,0.48,0.49,0.92);
  pad[0]->SetFillColor(0); 
  pad[1] = new TPad("pad_tr","The top-right pad",0.51,0.48,0.99,0.92); 
  pad[1]->SetFillColor(0);
  pad[2] = new TPad("pad_bl","The bottom-left pad",0.01,0.01,0.49,0.46); 
  pad[2]->SetFillColor(0);
  pad[3] = new TPad("pad_br","The bottom-right pad",0.51,0.01,0.99,0.46); 
  pad[3]->SetFillColor(0);

  for (unsigned int i=0; i<4; ++i) pad[i]->Draw();
                                                                                
  TLegend* leg = new TLegend(0.12,0.85,0.5,0.99);
  leg->SetFillColor(0);
  TFile* input_file = new TFile(chfile.Data(),"READONLY");
  TFile* input_fileref = new TFile(chfileref.Data(),"READONLY");
  bool first_plots_done = false;

  TString directory = DirectoryLast + "/BeforeCuts";

  TDirectory* dir_before = input_file->GetDirectory(directory);
  TDirectory* dirref_before = input_fileref->GetDirectory(directory);
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
//            h1->SetLineColor(kBlue);
//            h1->SetMarkerColor(kBlue);
            h1->SetMarkerStyle(21);
            h1->SetLineStyle(1);
            h1->SetLineWidth(3);
            h1->SetTitleSize(0.05,"X");
            h1->SetTitleSize(0.05,"Y");
            TString title=(TString)dir_before->Get(list_before->At(i+j)->GetName())->GetTitle();
            TString name=(TString)dir_before->Get(list_before->At(i+j)->GetName())->GetName(); 
            TString nameD =name+"_MC";
            h1->SetXTitle(title);
            h1->SetName(nameD);

            h1->SetYTitle("");
            h1->SetTitle(""); 
            h1->SetTitleOffset(0.85,"X");

            TH1D* hr = (TH1D*)dirref_before->Get(list_before->At(i+j)->GetName()); 
            hr->SetLineColor(kPink-4);
//            hr->SetLineStyle(2);
            hr->SetLineWidth(3);
            hr->SetTitleSize(0.05,"X");
            hr->SetTitleSize(0.05,"Y");
            hr->SetFillColor(kPink-4);
            hr->SetFillStyle(3001);
            hr->SetXTitle(title);
            TString nameMC =name+"_Data";
            h1->SetName(nameMC);

            hr->SetYTitle("");
            hr->SetTitle("");
            hr->SetTitleOffset(0.85,"X");


            if(logyFlag){
            h1->Draw("p,E");
            if(normalize) {hr->DrawNormalized("sames,hist",h1->Integral());}
            else{hr->Draw("sames,hist");}
            }
            else{
            if(normalize) {hr->DrawNormalized("hist",h1->Integral());}
            else{hr->Draw("hist");}
            h1->Draw("sames,p,E");
            }

            leg->Clear();
            leg->AddEntry(h1,"7TeV data","L");
            leg->AddEntry(hr,"MinimumBias MC","L");
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

  TString directory2 = DirectoryLast + "/LastCut";

  TDirectory* dir_lastcut = input_file->GetDirectory(directory2);
  TDirectory* dirref_lastcut = input_fileref->GetDirectory(directory2);
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
//            h1->SetLineColor(kBlue);
//            h1->SetMarkerColor(kBlue);
            h1->SetMarkerStyle(21);
            h1->SetLineWidth(3);
            h1->SetTitleSize(0.05,"X");
            h1->SetTitleSize(0.05,"Y");
            TString name=(TString)dir_lastcut->Get(list_lastcut->At(i+j)->GetName())->GetName();
            TString title=(TString)dir_lastcut->Get(list_lastcut->At(i+j)->GetName())->GetTitle();
            TString nameD=name+"_Data";
            h1->SetXTitle(title); 
            h1->SetName(nameD);
            h1->SetYTitle("");
            h1->SetTitle(""); 
            h1->SetTitleOffset(0.85,"X");

            TH1D* hr = (TH1D*)dirref_lastcut->Get(list_lastcut->At(i+j)->GetName()); 
            hr->SetLineColor(kAzure+5);
//            hr->SetLineStyle(2);
            hr->SetLineWidth(3);
            hr->SetTitleSize(0.05,"X");
            hr->SetTitleSize(0.05,"Y");
            hr->SetFillColor(kAzure+5);
            hr->SetFillStyle(3001);
            hr->SetXTitle(title);
            TString nameMC=name+"_Data";
            h1->SetName(nameMC);
            hr->SetYTitle("");
            hr->SetTitle("");
            hr->SetTitleOffset(0.85,"X");

            if(logyFlag){
            h1->Draw("p,E");
            if(normalize) {hr->DrawNormalized("sames,hist",h1->Integral());}
            else{hr->Draw("sames,hist");}
            }
            else{
            if(normalize) {hr->DrawNormalized("hist",h1->Integral());}
            else{hr->Draw("hist");}
            h1->Draw("sames,p,E");
            }


            

            leg->Clear();
            leg->AddEntry(h1,"7TeV data" ,"L");
            leg->AddEntry(hr,"MinimumBias MC","L");
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
