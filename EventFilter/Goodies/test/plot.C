#include "TROOT.h"
#include "TCanvas.h"
#include "TProfile.h"

float plot(int run, int maxbin){
  TString filename2="summary";
  filename2+=run;
  filename2+=".root";
  TString filename3="synopsis";
  filename3+=run;
  filename3+=".pdf";
  TFile *f2 = new TFile(filename2,"READONLY");
  f2->cd();
  TCanvas *c2 = new TCanvas();
  c2->Divide(2,2);
  TPad *pad1 = c2->cd(1);

  pad1->SetBorderMode(0);  
  
  c2->SetFrameBorderMode(0);

  TProfile *bf24_pxf = bf24->ProfileX();
  TProfile *bf20_pxf =bf20->ProfileX();
  TProfile *bf16_pxf =bf16->ProfileX();
  TProfile *bf11_pxf =bf11->ProfileX();
  TProfile *bf8_pxf =bf8->ProfileX();
  TProfile *bf7_pxf =bf7->ProfileX();
  TProfile *bf24_pxf = bf24->ProfileX();
  TProfile *bf20_pxf =bf20->ProfileX();
  TProfile *bf16_pxf =bf16->ProfileX();
  TProfile *bf11_pxf =bf11->ProfileX();
  TProfile *bf8_pxf =bf8->ProfileX();
  TProfile *bf7_pxf =bf7->ProfileX();
  bf24_pxf->SetMaximum(1.1);
  bf24_pxf->SetStats(kFALSE);
  bf24_pxf->SetBins(maxbin,0.,float(maxbin));
  bf24_pxf->Draw();
  bf20_pxf->SetMarkerColor(4);
  bf20_pxf->Draw("SAME");
  bf16_pxf->SetMarkerColor(3);
  bf16_pxf->Draw("SAME");
  bf11_pxf->SetMarkerColor(2);
  bf11_pxf->Draw("SAME");
  bf8_pxf->SetMarkerColor(5);
  bf8_pxf->Draw("SAME");
  bf7_pxf->SetMarkerColor(6);
  bf7_pxf->Draw("SAME");
  
  c2->cd(2)->SetBorderMode(0);
  TProfile *rt24_pxf = rt24->ProfileX();
  TProfile *rt20_pxf =rt20->ProfileX();
  TProfile *rt16_pxf =rt16->ProfileX();
  TProfile *rt11_pxf =rt11->ProfileX();
  TProfile *rt8_pxf =rt8->ProfileX();
  TProfile *rt7_pxf =rt7->ProfileX();
  rt24_pxf->SetMaximum(170.);
  rt24_pxf->SetStats(kFALSE);
  rt24_pxf->SetMinimum(40.);
  rt24_pxf->SetBins(maxbin,0.,float(maxbin));
  rt24_pxf->Draw();
  rt20_pxf->SetMarkerColor(4);
  rt20_pxf->Draw("SAME");
  rt16_pxf->SetMarkerColor(3);
  rt16_pxf->Draw("SAME");
  rt11_pxf->SetMarkerColor(2);
  rt11_pxf->Draw("SAME");
  rt8_pxf->SetMarkerColor(5);
  rt8_pxf->Draw("SAME");
  rt7_pxf->SetMarkerColor(6);
  rt7_pxf->Draw("SAME");

  TPad *pad3 = c2->cd(3);
  pad3->SetBorderMode(0);
  TProfile *et24_pxf = et24->ProfileX();
  TProfile *et20_pxf =et20->ProfileX();
  TProfile *et16_pxf =et16->ProfileX();
  TProfile *et11_pxf =et11->ProfileX();
  TProfile *et8_pxf =et8->ProfileX();
  TProfile *et7_pxf =et7->ProfileX();
  et24_pxf->SetMaximum(0.110);
  et24_pxf->SetStats(kFALSE);
  et24_pxf->SetMinimum(0.05);
  et24_pxf->SetBins(maxbin,0.,float(maxbin));
  et24_pxf->Draw();


  et20_pxf->SetLineColor(4);
  et16_pxf->SetLineColor(3);
  et11_pxf->SetLineColor(2);
  et8_pxf->SetLineColor(5);
  et7_pxf->SetLineColor(6);


  et20_pxf->SetMarkerColor(4);
  et20_pxf->Draw("SAME");
  et16_pxf->SetMarkerColor(3);
  et16_pxf->Draw("SAME");
  et11_pxf->SetMarkerColor(2);
  et11_pxf->Draw("SAME");
  et8_pxf->SetMarkerColor(5);
  et8_pxf->Draw("SAME");
  et7_pxf->SetMarkerColor(6);
  et7_pxf->Draw("SAME");
  pad3->BuildLegend();

  TPad *pad4 = c2->cd(4);
  pad4->SetBorderMode(0);

//   et24_pxf->SetMaximum(0.09);
//   et24_pxf->SetMinimum(0.05);
//   et24_pxf->SetBins(maxbin,0.,float(maxbin));
//   et24_pxf->Draw();
//   et20_pxf->SetMarkerColor(4);
//   et20_pxf->Draw("SAME");
//   et16_pxf->SetMarkerColor(3);
//   et16_pxf->Draw("SAME");
//   et11_pxf->SetMarkerColor(2);
//   et11_pxf->Draw("SAME");
  TH1F *ratio1 = new TH1F(*set7);
  ratio1->Divide(set11);
  ratio1->SetMarkerColor(2);
  ratio1->SetLineColor(2);
  ratio1->SetBins(maxbin,0.,float(maxbin));
  ratio1->SetStats(kFALSE);
  ratio1->Draw();
  TH1F *ratio2 = new TH1F(*set11);
  ratio2.Divide(set16);
  ratio2->SetMarkerColor(3);
  ratio2->Draw("SAME");
  c2->Print(filename3);
  TLine *l1 = new TLine(0.,1.4,float(maxbin),1.4);
  TLine *l2 = new TLine(0.,1.,float(maxbin),1.);
  l1->Draw();
  l2->Draw();
  pad4->BuildLegend()
}
