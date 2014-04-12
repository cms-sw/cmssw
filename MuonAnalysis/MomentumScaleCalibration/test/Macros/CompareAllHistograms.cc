#include <iostream>

#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "TObject.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TString.h"

void CompareAllHistograms(const TString & fileName1, const TString & fileName2)
{
  TFile file1(fileName1);
  TFile file2(fileName2);

  TH1 *histo1, *histo2;
  TKey *key;
  TIter nextkey(file1.GetListOfKeys());

  TFile outputFile("ComparedHistograms.root", "RECREATE");
  outputFile.cd();

  while( key = (TKey*)nextkey() ) {
    TKey * obj = (TKey*)(key->ReadObj());
    if( (obj->IsA()->InheritsFrom("TH1F")) || (obj->IsA()->InheritsFrom("TH1D")) ) {
      histo1 = (TH1*)obj; 
      histo2 = (TH1*)file2.FindObjectAny(histo1->GetName());
      std::cout << histo1->GetName() << std::endl;
      TCanvas canvas(TString(histo1->GetName())+"_canvas", histo1->GetName(), 1000, 800);
      canvas.cd();
      histo1->Draw("E1");
      histo2->SetLineColor(kRed);
      histo2->Sumw2();
      histo2->Scale(histo1->Integral()/histo2->Integral());
      // histo2->Scale(histo1->GetEntries()/histo2->GetEntries());
      histo2->Draw("E1same");
      canvas.Write();
      // c0->Modified();
      // c0->Update(); 
    }
  }
  outputFile.Write();
  outputFile.Close();
}
