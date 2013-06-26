void mbplot(int flag=0){
  gROOT->LoadMacro("mbtkStyle.C");
  setTDRStyle();

  if (((flag)%2) == 1) {
    TFile *CMSSWFile1 = new TFile("matbdgtkC.root");
    TProfile *histCMSSW1 = (TProfile*) CMSSWFile1->Get("10");
    TH1D *c1 = histCMSSW1->ProjectionX();

    TFile *OSCARFile1 = new TFile("matbdgtkO.root");
    TProfile *histOSCAR1 = (TProfile*) OSCARFile1->Get("10");
    TH1D *o1 = histOSCAR1->ProjectionX();

    TCanvas *mbtkplot = new TCanvas("Tracker MB","Tracker MB",1);

    c1->SetLineStyle(1);
    c1->SetLineWidth(3);
    c1->SetLineColor(2);
    c1->SetFillColor(5);
    c1->SetFillStyle(3013);
    c1->SetMinimum(0.);
    c1->SetMaximum(1.7);
    c1->GetXaxis()->SetRangeUser(0.0, 2.5);
    c1->GetXaxis()->SetTitle("#eta");
    c1->GetYaxis()->SetTitle("X/X_{0}");
    c1->Draw("HIST");

    o1->SetLineStyle(2);
    o1->SetLineWidth(3);
    o1->SetLineColor(4);
    o1->SetFillColor(7);
    o1->SetFillStyle(3013);
    o1->Draw("HIST same");

    leg1 = new TLegend(0.55,0.35,0.85,0.45);
    leg1->AddEntry(c1,"CMSSW Geometry","F");
    leg1->AddEntry(o1,"OSCAR Geometry","F");
    leg1->SetHeader("Complete Tracker");
    leg1->Draw();

    mbtkplot->SaveAs("mbtkfig.eps");
  }

  if (((flag/2)%2) == 1) {
    TFile *CMSSWFile1 = new TFile("matbdgpxC.root");
    TProfile *histCMSSW1 = (TProfile*) CMSSWFile1->Get("10");
    TH1D *c1 = histCMSSW1->ProjectionX();

    TFile *OSCARFile1 = new TFile("matbdgpxO.root");
    TProfile *histOSCAR1 = (TProfile*) OSCARFile1->Get("10");
    TH1D *o1 = histOSCAR1->ProjectionX();

    TCanvas *mbpxplot = new TCanvas("Pixel MB","Pixel MB",1);

    c1->SetLineStyle(1);
    c1->SetLineWidth(3);
    c1->SetLineColor(2);
    c1->SetFillColor(5);
    c1->SetFillStyle(3013);
    c1->SetMinimum(0.);
    c1->SetMaximum(1.7);
    c1->GetXaxis()->SetRangeUser(0.0, 2.5);
    c1->GetXaxis()->SetTitle("#eta");
    c1->GetYaxis()->SetTitle("X/X_{0}");
    c1->Draw("HIST");

    o1->SetLineStyle(2);
    o1->SetLineWidth(3);
    o1->SetLineColor(4);
    o1->SetFillColor(7);
    o1->SetFillStyle(3013);
    o1->Draw("HIST same");

    leg1 = new TLegend(0.55,0.4,0.85,0.5);
    leg1->AddEntry(c1,"CMSSW Geometry","F");
    leg1->AddEntry(o1,"OSCAR Geometry","F");
    leg1->SetHeader("Pixel Detectors");
    leg1->Draw();

    mbpxplot->SaveAs("mbpxfig.eps");
  }

  if (((flag/4)%2) == 1) {
    TFile *CMSSWFile1 = new TFile("matbdgtibC.root");
    TProfile *histCMSSW1 = (TProfile*) CMSSWFile1->Get("10");
    TH1D *c1 = histCMSSW1->ProjectionX();

    TFile *OSCARFile1 = new TFile("matbdgtibO.root");
    TProfile *histOSCAR1 = (TProfile*) OSCARFile1->Get("10");
    TH1D *o1 = histOSCAR1->ProjectionX();

    TCanvas *mbtibplot = new TCanvas("TIB MB","TIB MB",1);

    c1->SetLineStyle(1);
    c1->SetLineWidth(3);
    c1->SetLineColor(2);
    c1->SetFillColor(5);
    c1->SetFillStyle(3013);
    c1->SetMinimum(0.);
    c1->SetMaximum(1.7);
    c1->GetXaxis()->SetRangeUser(0.0, 2.5);
    c1->GetXaxis()->SetTitle("#eta");
    c1->GetYaxis()->SetTitle("X/X_{0}");
    c1->Draw("HIST");

    o1->SetLineStyle(2);
    o1->SetLineWidth(3);
    o1->SetLineColor(4);
    o1->SetFillColor(7);
    o1->SetFillStyle(3013);
    o1->Draw("HIST same");

    leg1 = new TLegend(0.55,0.4,0.85,0.5);
    leg1->AddEntry(c1,"CMSSW Geometry","F");
    leg1->AddEntry(o1,"OSCAR Geometry","F");
    leg1->SetHeader("TIB");
    leg1->Draw();

    mbtibplot->SaveAs("mbtibfig.eps");
  }

  if (((flag/8)%2) == 1) {
    TFile *CMSSWFile1 = new TFile("matbdgtidC.root");
    TProfile *histCMSSW1 = (TProfile*) CMSSWFile1->Get("10");
    TH1D *c1 = histCMSSW1->ProjectionX();

    TFile *OSCARFile1 = new TFile("matbdgtidO.root");
    TProfile *histOSCAR1 = (TProfile*) OSCARFile1->Get("10");
    TH1D *o1 = histOSCAR1->ProjectionX();

    TCanvas *mbtidplot = new TCanvas("TID MB","TID MB",1);

    c1->SetLineStyle(1);
    c1->SetLineWidth(3);
    c1->SetLineColor(2);
    c1->SetFillColor(5);
    c1->SetFillStyle(3013);
    c1->SetMinimum(0.);
    c1->SetMaximum(1.7);
    c1->GetXaxis()->SetRangeUser(0.0, 2.5);
    c1->GetXaxis()->SetTitle("#eta");
    c1->GetYaxis()->SetTitle("X/X_{0}");
    c1->Draw("HIST");

    o1->SetLineStyle(2);
    o1->SetLineWidth(3);
    o1->SetLineColor(4);
    o1->SetFillColor(7);
    o1->SetFillStyle(3013);
    o1->Draw("HIST same");

    leg1 = new TLegend(0.55,0.4,0.85,0.5);
    leg1->AddEntry(c1,"CMSSW Geometry","F");
    leg1->AddEntry(o1,"OSCAR Geometry","F");
    leg1->SetHeader("TID");
    leg1->Draw();

    mbtidplot->SaveAs("mbtidfig.eps");
  }

  if (((flag/16)%2) == 1) {
    TFile *CMSSWFile1 = new TFile("matbdgtobC.root");
    TProfile *histCMSSW1 = (TProfile*) CMSSWFile1->Get("10");
    TH1D *c1 = histCMSSW1->ProjectionX();

    TFile *OSCARFile1 = new TFile("matbdgtobO.root");
    TProfile *histOSCAR1 = (TProfile*) OSCARFile1->Get("10");
    TH1D *o1 = histOSCAR1->ProjectionX();

    TCanvas *mbtobplot = new TCanvas("TOB MB","TOB MB",1);

    c1->SetLineStyle(1);
    c1->SetLineWidth(3);
    c1->SetLineColor(2);
    c1->SetFillColor(5);
    c1->SetFillStyle(3013);
    c1->SetMinimum(0.);
    c1->SetMaximum(0.5);
    c1->GetXaxis()->SetRangeUser(0.0, 2.5);
    c1->GetXaxis()->SetTitle("#eta");
    c1->GetYaxis()->SetTitle("X/X_{0}");
    c1->Draw("HIST");

    o1->SetLineStyle(2);
    o1->SetLineWidth(3);
    o1->SetLineColor(4);
    o1->SetFillColor(7);
    o1->SetFillStyle(3013);
    o1->Draw("HIST same");

    leg1 = new TLegend(0.55,0.4,0.85,0.5);
    leg1->AddEntry(c1,"CMSSW Geometry","F");
    leg1->AddEntry(o1,"OSCAR Geometry","F");
    leg1->SetHeader("TOB");
    leg1->Draw();

    mbtobplot->SaveAs("mbtobfig.eps");
  }

  if (((flag/32)%2) == 1) {
    TFile *CMSSWFile1 = new TFile("matbdgtecC.root");
    TProfile *histCMSSW1 = (TProfile*) CMSSWFile1->Get("10");
    TH1D *c1 = histCMSSW1->ProjectionX();

    TFile *OSCARFile1 = new TFile("matbdgtecO.root");
    TProfile *histOSCAR1 = (TProfile*) OSCARFile1->Get("10");
    TH1D *o1 = histOSCAR1->ProjectionX();

    TCanvas *mbtecplot = new TCanvas("TEC MB","TEC MB",1);

    c1->SetLineStyle(1);
    c1->SetLineWidth(3);
    c1->SetLineColor(2);
    c1->SetFillColor(5);
    c1->SetFillStyle(3013);
    c1->SetMinimum(0.);
    c1->SetMaximum(1.7);
    c1->GetXaxis()->SetRangeUser(0.0, 2.5);
    c1->GetXaxis()->SetTitle("#eta");
    c1->GetYaxis()->SetTitle("X/X_{0}");
    c1->Draw("HIST");

    o1->SetLineStyle(2);
    o1->SetLineWidth(3);
    o1->SetLineColor(4);
    o1->SetFillColor(7);
    o1->SetFillStyle(3013);
    o1->Draw("HIST same");

    leg1 = new TLegend(0.55,0.4,0.85,0.5);
    leg1->AddEntry(c1,"CMSSW Geometry","F");
    leg1->AddEntry(o1,"OSCAR Geometry","F");
    leg1->SetHeader("TEC");
    leg1->Draw();

    mbtecplot->SaveAs("mbtecfig.eps");
  }
}
