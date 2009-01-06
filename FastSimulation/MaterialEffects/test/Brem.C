TCanvas* BremComparison;

void DrawComparison(TH1F* fast, TH1F* full, int colfast=2, int colfull=2, int style=22) {

   // Draw Fast
   fast->SetStats(0);
   fast->SetTitle("");
   fast->SetLineWidth(colfast);
   fast->SetMarkerColor(colfast);
   fast->SetMarkerStyle(style);
   fast->GetYaxis()->SetTitleOffset(1.6);
   if ( fast->GetName() == "EGammaFast" )  
     fast->SetXTitle("Energy (GeV)");
   else if ( fast->GetName() == "FEGammaFast" ) 
     fast->SetXTitle("Energy Fraction");
   if ( fast->GetName() == "NGammaFast" || 
	fast->GetName() == "NGammaMinFast" )  
     fast->SetXTitle("Photons per electrons");
   else
     fast->SetXTitle("#eta");
   fast->SetYTitle("Nb. of photons");
   fast->Draw("sameerro");

   // Draw full
   full->SetStats(0);
   full->SetTitle("");
   full->SetLineColor(colfull);
   full->SetLineWidth(2);
   full->Draw("same");

}

void Comparison(TH1F* fast, TH1F* full) {
  BremComparison->Delete();
  //  fast->SetMaximum(maximum);
  BremComparison = new TCanvas("Brem","BremComparison",150,150,800,600);
  DrawComparison(fast,full,2,4);
  double nfast = fast->GetEntries();
  double nfull = full->GetEntries();
  cout << "Full/Fast/ratio = " << nfull << " " << nfast << " " << nfull/nfast << endl;
  //  BremComparison->SaveAs(Form("%s_Test.eps", fast->GetName()));
  //  BremComparison->SaveAs(Form("%s_Test.gif", fast->GetName()));
}

void Ratio(TH1F* fast, TH1F* full) {
  BremComparison->Delete();
  //  fast->SetMaximum(maximum);
  BremComparison = new TCanvas("Brem","BremComparison",150,150,800,600);
  BremComparison->SetGridx();
  BremComparison->SetGridy();
  TH1F* ratio = (TH1F*)full->Clone();
  ratio->Divide(fast);
  ratio->SetLineColor(4);
  ratio->SetLineWidth(2);
  ratio->SetTitle(Form("%s_Ratio",fast->GetName()));
  //  ratio->Fit("pol0");
  ratio->Draw();
  //  BremComparison->SaveAs(Form("%s_Ratio_Test.eps", fast->GetName()));
  //  BremComparison->SaveAs(Form("%s_Ratio_Test.gif", fast->GetName()));
}

void Brem() {

  gDirectory->cd("DQMData");
  BremComparison = new TCanvas("Brem","BremComparison",150,150,800,600);
  //  TrackerFast->SetMaximum(24000);
  TrackerFast->SetMaximum(9000);
  TrackerFast->SetMinimum(0);
  TLegend* pixel = new TLegend(0.4,0.17,0.47,0.20);
  pixel->AddEntry("Pixels ","Pixels ","");
  TLegend* inner = new TLegend(0.28,0.37,0.42,0.40);
  inner->AddEntry("Inner Tracker  ","Inner Tracker  ","");
  TLegend* outer = new TLegend(0.30,0.57,0.44,0.60);
  outer->AddEntry("Outer Tracker  ","Outer Tracker  ","");

  DrawComparison(TrackerFast,TrackerFull);
  DrawComparison(OuterFast,OuterFull,4,4);
  DrawComparison(InnerFast,InnerFull,3,3);
  DrawComparison(PixelFast,PixelFull,1,1);
  //  DrawComparison(BPFast,BPFull,5,5);

  pixel->Draw();
  inner->Draw();
  outer->Draw();

  BremComparison->SaveAs("Test.eps");
  BremComparison->SaveAs("Test.gif");

}
