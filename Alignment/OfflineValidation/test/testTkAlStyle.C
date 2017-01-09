#include "TCanvas.h"
#include "TH1.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TROOT.h"

#include "../plugins/TkAlStyle.cc"


void testTkAlStyle() {
  gROOT->ProcessLine(".L ../plugins/TkAlStyle.cc+");
  TkAlStyle::set(PRELIMINARY);	// set publication status

  TCanvas* can = new TCanvas("can","can",500,500);
  can->cd();


  // Create dummy histograms representing validation plots,
  // e.g. DMR plots, for a particular alignment object, using
  // line style accordingly
  TH1* h1 = new TH1D("h1",";x title;y title",100,-10,10);
  h1->FillRandom("gaus",1000);
  h1->SetLineColor(TkAlStyle::color(IDEALAlign));
  h1->SetLineStyle(TkAlStyle::style(IDEALAlign));
  h1->GetYaxis()->SetRangeUser(0,110);

  TH1* h2 = new TH1D("h2",";x title;y title",100,-10,10);
  h2->FillRandom("gaus",500);
  h2->SetLineColor(TkAlStyle::color(CRAFTAlign));
  h2->SetLineStyle(TkAlStyle::style(CRAFTAlign));
  h2->GetYaxis()->SetRangeUser(0,110);

  h1->Draw();
  h2->Draw("same");


  // Add a title that specifies the data-taking era
  // (title specifies also the publication label "CMS Preliminary"
  // etc. according to the status set above)
  TPaveText* title = TkAlStyle::standardTitle(CRAFT15);
  title->Draw("same");


  // Add a legend at the top left with 2 entries stretching
  // over 60% of the pad's width. Legend labels depend on
  // the alignment object.
  TLegend* leg = TkAlStyle::legend("top left",2,0.6);
  leg->AddEntry(h1,toTString(IDEALAlign),"L");
  leg->AddEntry(h2,toTString(CRAFTAlign),"L");
  leg->Draw("same");

  gPad->RedrawAxis();
  can->SaveAs("test.pdf");
}
