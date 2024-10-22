#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TObjArray.h"
#include "TObject.h"
#include "TClass.h"
#include <iostream>
#include <string>
#include "TPaveText.h"

#include <fstream>  // std::ofstream

//*****************************************************//
Double_t getMaximum(TObjArray *array)
//*****************************************************//
{
  Double_t theMaximum = (static_cast<TH1 *>(array->At(0)))->GetMaximum();
  for (Int_t i = 0; i < array->GetSize(); i++) {
    if ((static_cast<TH1 *>(array->At(i)))->GetMaximum() > theMaximum) {
      theMaximum = (static_cast<TH1 *>(array->At(i)))->GetMaximum();
      //cout<<"i= "<<i<<" theMaximum="<<theMaximum<<endl;
    }
  }
  return theMaximum;
}

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   MAIN
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
*/

//*****************************************************//
void compareAll(TString file1, TString file2, TString leg1, TString leg2)
//*****************************************************//
{
  TFile *f1 = TFile::Open(file1);
  TFile *f2 = TFile::Open(file2);

  f1->cd();

  int i = 0;
  int j = 0;

  TCanvas dummyC;
  dummyC.Print("diff.pdf[");

  std::ofstream ofs("check.txt", std::ofstream::out);

  TIter nextkey(gDirectory->GetListOfKeys());
  while (TKey *key = (TKey *)nextkey()) {
    //std::cout << "i: " << i << std::endl;
    ++i;
    TObject *obj = key->ReadObj();
    std::string name = obj->GetName();
    std::cout << "name: " << name << std::endl;
    if (name.find("Residuals") != string::npos) {
      std::cout << "skipping:" << name << "folder" << std::endl;
      continue;
    }
    if (obj->IsA()->InheritsFrom("TDirectory")) {
      f1->cd((name).c_str());
      TIter nextkey(gDirectory->GetListOfKeys());
      while (key = (TKey *)nextkey()) {
        obj = key->ReadObj();
        if (obj->IsA()->InheritsFrom("TH1")) {
          TH1 *h = (TH1 *)obj;
          TString fullpath = name + "/" + h->GetName();
          //std::cout << "j: " << j << " "<< h->GetName() <<" "<< fullpath << std::endl;
          ++j;
          if (TString(h->GetName()).Contains("Charge_Vs_Index"))
            continue;
          TCanvas *c1 = new TCanvas(h->GetName(), h->GetName(), 800, 600);
          c1->cd();

          TPad *pad1;
          if (obj->IsA()->InheritsFrom("TH2")) {
            pad1 = new TPad("pad1", "pad1", 0, 0., 0.5, 1.0);
          } else {
            pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
            pad1->SetBottomMargin(0.01);  // Upper and lower plot are joined
          }
          //	  pad1->SetGridx();         // Vertical grid
          pad1->Draw();  // Draw the upper pad: pad1
          pad1->cd();    // pad1 becomes the current pad

          h->SetMarkerColor(kRed);
          h->SetStats(kFALSE);
          h->SetLineColor(kRed);
          h->SetMarkerStyle(kOpenSquare);
          h->GetXaxis()->SetLabelOffset(0.2);

          h->GetYaxis()->SetTitleSize(0.05);
          //h->GetYaxis()->SetTitleFont(43);
          h->GetYaxis()->SetTitleOffset(0.8);

          TH1 *h2 = (TH1 *)f2->Get(fullpath.Data());
          h2->SetStats(kFALSE);

          if (h2 == nullptr) {
            std::cout << "WARNING!!!!!! " << fullpath << " does NOT exist in second file!!!!!" << std::endl;
            continue;
          }

          h2->SetMarkerColor(kBlue);
          h2->SetLineColor(kBlue);
          h2->SetMarkerStyle(kOpenCircle);
          h2->GetXaxis()->SetLabelOffset(0.2);
          h2->GetYaxis()->SetTitleOffset(0.8);

          TObjArray *arrayHistos = new TObjArray();
          arrayHistos->Expand(2);
          arrayHistos->Add(h);
          arrayHistos->Add(h2);

          if (!obj->IsA()->InheritsFrom("TH2")) {
            Double_t theMaximum = getMaximum(arrayHistos);
            h->GetYaxis()->SetRangeUser(0., theMaximum * 1.30);
            h->Draw();
            h2->Draw("same");
          } else {
            c1->cd();
            pad1->cd();
            h->SetMarkerSize(0.1);
            h->Draw("colz");

            TLegend *lego = new TLegend(0.12, 0.80, 0.29, 0.88);
            lego->SetFillColor(10);
            lego->SetTextSize(0.042);
            lego->SetTextFont(42);
            lego->SetFillColor(10);
            lego->SetLineColor(10);
            lego->SetShadowColor(10);
            lego->AddEntry(h, leg1.Data());
            lego->Draw();

            c1->cd();
            TPad *pad2 = new TPad("pad2", "pad2", 0.5, 0., 1.0, 1.0);
            pad2->Draw();
            pad2->cd();
            h2->SetMarkerSize(0.1);
            h2->Draw("colz");

            TLegend *lego2 = new TLegend(0.12, 0.80, 0.29, 0.88);
            lego2->SetFillColor(10);
            lego2->SetTextSize(0.042);
            lego2->SetTextFont(42);
            lego2->SetFillColor(10);
            lego2->SetLineColor(10);
            lego2->SetShadowColor(10);
            lego2->AddEntry(h2, leg2.Data());
            lego2->Draw("same");
          }

          TString savename = fullpath.ReplaceAll("/", "_");
          double ksProb = 0;

          // lower plot will be in pad
          if (!obj->IsA()->InheritsFrom("TH2")) {
            TLegend *lego = new TLegend(0.12, 0.80, 0.29, 0.88);
            lego->SetFillColor(10);
            lego->SetTextSize(0.042);
            lego->SetTextFont(42);
            lego->SetFillColor(10);
            lego->SetLineColor(10);
            lego->SetShadowColor(10);
            lego->AddEntry(h, leg1.Data());
            lego->AddEntry(h2, leg2.Data());

            lego->Draw("same");

            c1->cd();  // Go back to the main canvas before defining pad2

            TPad *pad2 = new TPad("pad2", "pad2", 0, 0.05, 1, 0.3);
            pad2->SetTopMargin(0.01);
            pad2->SetBottomMargin(0.35);
            pad2->SetGridy();  // horizontal grid
            pad2->Draw();
            pad2->cd();  // pad2 becomes the current pad

            // Define the ratio plot
            TH1F *h3 = (TH1F *)h->Clone("h3");
            h3->SetLineColor(kBlack);
            h3->SetMarkerColor(kBlack);
            h3->SetTitle("");
            h3->SetMinimum(0.55);  // Define Y ..
            h3->SetMaximum(1.55);  // .. range
            h3->SetStats(0);       // No statistics on lower plot
            h3->Divide(h2);
            h3->SetMarkerStyle(20);
            h3->Draw("ep");  // Draw the ratio plot

            // Y axis ratio plot settings
            h3->GetYaxis()->SetTitle("ratio");
            h3->GetYaxis()->SetNdivisions(505);
            h3->GetYaxis()->SetTitleSize(20);
            h3->GetYaxis()->SetTitleFont(43);
            h3->GetYaxis()->SetTitleOffset(1.2);
            h3->GetYaxis()->SetLabelFont(43);  // Absolute font size in pixel (precision 3)
            h3->GetYaxis()->SetLabelSize(15);

            // X axis ratio plot settings
            h3->GetXaxis()->SetTitleSize(20);
            h3->GetXaxis()->SetTitleFont(43);
            h3->GetXaxis()->SetTitleOffset(4.);
            h3->GetXaxis()->SetLabelFont(43);  // Absolute font size in pixel (precision 3)
            h3->GetXaxis()->SetLabelSize(15);
            h3->GetXaxis()->SetLabelOffset(0.02);
          }

          // avoid spurious false positive due to empty histogram
          if (h->GetEntries() == 0 && h2->GetEntries() == 0) {
            ofs << "histogram # " << j << ": " << fullpath << " |has zero entries in both files: " << std::endl;
            delete c1;
            delete h;
            delete h2;
            continue;
          }
          // avoid doing k-test on histograms with different binning
          if (h->GetXaxis()->GetXmax() != h2->GetXaxis()->GetXmax() ||
              h->GetXaxis()->GetXmin() != h2->GetXaxis()->GetXmin()) {
            ofs << "histogram # " << j << ": " << fullpath << " |mismatched bins!!!!!!: " << std::endl;
            delete c1;
            delete h;
            delete h2;
            continue;
          }

          ksProb = h->KolmogorovTest(h2);
          //if(ksProb!=1.){
          //c1->SaveAs(savename+".pdf");
          TPaveText ksPt(0, 0.02, 0.35, 0.043, "NDC");
          ksPt.SetBorderSize(0);
          ksPt.SetFillColor(0);
          ksPt.AddText(Form("P(KS)=%g, ered %g eblue %g", ksProb, h->GetEntries(), h2->GetEntries()));
          if (!obj->IsA()->InheritsFrom("TH2")) {
            ksPt.SetTextSize(0.1);
            ksPt.Draw();
          } else {
            ksPt.SetTextSize(0.03);
            ksPt.Draw();
          }
          c1->Print("diff.pdf");
          std::cout << "histogram # " << j << ": " << fullpath << " |kolmogorov: " << ksProb << std::endl;
          ofs << "histogram # " << j << ": " << fullpath << " |kolmogorov: " << ksProb << std::endl;
          // }

          delete c1;
          delete h;
          delete h2;
        }
      }
    } else if (obj->IsA()->InheritsFrom("TH1")) {
      obj = key->ReadObj();
      TH1 *h = (TH1 *)obj;
      TString fullpath = (TString)h->GetName();
      //std::cout << "j: " << j << " "<< h->GetName() <<" "<< fullpath << std::endl;
      ++j;
      if (obj->IsA()->InheritsFrom("TH2"))
        continue;
      TCanvas *c1 = new TCanvas(h->GetName(), h->GetName(), 800, 600);
      c1->cd();
      TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
      pad1->SetBottomMargin(0.01);  // Upper and lower plot are joined
      //	  pad1->SetGridx();         // Vertical grid
      pad1->Draw();  // Draw the upper pad: pad1
      pad1->cd();    // pad1 becomes the current pad

      h->SetMarkerColor(kRed);
      h->SetStats(kFALSE);
      h->SetLineColor(kRed);
      h->SetMarkerStyle(kOpenSquare);
      h->GetXaxis()->SetLabelOffset(0.2);

      h->GetYaxis()->SetTitleSize(0.05);
      //h->GetYaxis()->SetTitleFont(43);
      h->GetYaxis()->SetTitleOffset(0.8);

      TH1 *h2 = (TH1 *)f2->Get(fullpath.Data());
      h2->SetStats(kFALSE);

      if (h2 == nullptr) {
        std::cout << "WARNING!!!!!! " << fullpath << " does NOT exist in second file!!!!!" << std::endl;
        continue;
      }

      h2->SetMarkerColor(kBlue);
      h2->SetLineColor(kBlue);
      h2->SetMarkerStyle(kOpenCircle);
      h2->GetXaxis()->SetLabelOffset(0.2);
      h2->GetYaxis()->SetTitleOffset(0.8);

      TObjArray *arrayHistos = new TObjArray();
      arrayHistos->Expand(2);
      arrayHistos->Add(h);
      arrayHistos->Add(h2);

      Double_t theMaximum = getMaximum(arrayHistos);
      h->GetYaxis()->SetRangeUser(0., theMaximum * 1.30);

      h->Draw();
      h2->Draw("same");
      TString savename = fullpath.ReplaceAll("/", "_");
      double ksProb = 0;

      TLegend *lego = new TLegend(0.12, 0.80, 0.29, 0.88);
      lego->SetFillColor(10);
      lego->SetTextSize(0.042);
      lego->SetTextFont(42);
      lego->SetFillColor(10);
      lego->SetLineColor(10);
      lego->SetShadowColor(10);
      lego->AddEntry(h, leg1.Data());
      lego->AddEntry(h2, leg2.Data());

      lego->Draw("same");

      // lower plot will be in pad
      c1->cd();  // Go back to the main canvas before defining pad2
      TPad *pad2 = new TPad("pad2", "pad2", 0, 0.05, 1, 0.3);
      pad2->SetTopMargin(0.01);
      pad2->SetBottomMargin(0.35);
      pad2->SetGridy();  // horizontal grid
      pad2->Draw();
      pad2->cd();  // pad2 becomes the current pad

      // Define the ratio plot
      TH1F *h3 = (TH1F *)h->Clone("h3");
      h3->SetLineColor(kBlack);
      h3->SetMarkerColor(kBlack);
      h3->SetTitle("");
      h3->SetMinimum(0.55);  // Define Y ..
      h3->SetMaximum(1.55);  // .. range
      h3->SetStats(0);       // No statistics on lower plot
      h3->Divide(h2);
      h3->SetMarkerStyle(20);
      h3->Draw("ep");  // Draw the ratio plot

      // Y axis ratio plot settings
      h3->GetYaxis()->SetTitle("ratio");
      h3->GetYaxis()->SetNdivisions(505);
      h3->GetYaxis()->SetTitleSize(20);
      h3->GetYaxis()->SetTitleFont(43);
      h3->GetYaxis()->SetTitleOffset(1.2);
      h3->GetYaxis()->SetLabelFont(43);  // Absolute font size in pixel (precision 3)
      h3->GetYaxis()->SetLabelSize(15);

      // X axis ratio plot settings
      h3->GetXaxis()->SetTitleSize(20);
      h3->GetXaxis()->SetTitleFont(43);
      h3->GetXaxis()->SetTitleOffset(4.);
      h3->GetXaxis()->SetLabelFont(43);  // Absolute font size in pixel (precision 3)
      h3->GetXaxis()->SetLabelSize(15);
      h3->GetXaxis()->SetLabelOffset(0.02);

      // avoid spurious false positive due to empty histogram
      if (h->GetEntries() == 0 && h2->GetEntries() == 0) {
        ofs << "histogram # " << j << ": " << fullpath << " |has zero entries in both files: " << std::endl;
        delete c1;
        delete h;
        delete h2;
        continue;
      }
      // avoid doing k-test on histograms with different binning
      if (h->GetXaxis()->GetXmax() != h2->GetXaxis()->GetXmax() ||
          h->GetXaxis()->GetXmin() != h2->GetXaxis()->GetXmin()) {
        ofs << "histogram # " << j << ": " << fullpath << " |mismatched bins!!!!!!: " << std::endl;
        delete c1;
        delete h;
        delete h2;
        continue;
      }

      ksProb = h->KolmogorovTest(h2);
      //if(ksProb!=1.){
      //c1->SaveAs(savename+".pdf");
      TPaveText ksPt(0, 0.02, 0.35, 0.043, "NDC");
      ksPt.SetBorderSize(0);
      ksPt.SetFillColor(0);
      ksPt.AddText(Form("P(KS)=%g, ered %g eblue %g", ksProb, h->GetEntries(), h2->GetEntries()));
      ksPt.SetTextSize(0.1);
      ksPt.Draw();
      c1->Print("diff.pdf");
      std::cout << "histogram # " << j << ": " << fullpath << " |kolmogorov: " << ksProb << std::endl;
      ofs << "histogram # " << j << ": " << fullpath << " |kolmogorov: " << ksProb << std::endl;
      // }

      delete c1;
      delete h;
      delete h2;
    }
  }  //while
  f1->Close();
  f2->Close();
  dummyC.Print("diff.pdf]");

  ofs.close();
}
