#include <iostream>
#include <TFile.h>
#include <TH1F.h>
#include <TList.h>
#include <TCanvas.h>
#include <TROOT.h>

void histo_check (Int_t min_height=4) {
  gROOT->SetStyle("Plain");

  TString dir="tmp";
  TString tag="example";
  TFile f("/"+dir+"/"+tag+"/"+tag+".root");
  f.cd("adc");
  TList *list=gDirectory->GetListOfKeys();
  Int_t stop=list->LastIndex();

  TCanvas c1("c1");

  Int_t num=0;
  Int_t num_limit=1000;
  Int_t i_print=5000;

  TString ps_file="/"+dir+"/"+tag+"/histos.ps";

  c1.Print(ps_file+"[");
  for (Int_t i=0;i<=stop;i++) {
    //if (!(i%i_print)) cout << i << endl;

    TString str=list->At(i)->GetName();
    TH1F *h=(TH1F*)f.Get("adc/"+str)->Clone();

    Float_t height=h->GetBinContent(h->GetMaximumBin());
    if (height>min_height) {
      h->Draw();
      cout << h->GetName() << endl;
      c1.Print(ps_file);
      num++;
    }
    //if (height==0.0) {
    //  cout << h->GetName() << endl;
    //}
    delete h;
    if (num>num_limit) break;
  }
  c1.Print(ps_file+"]");
}
