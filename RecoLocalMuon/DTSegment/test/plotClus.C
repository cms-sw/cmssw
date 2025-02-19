{
  gROOT->LoadMacro("macros.C");     // Load service macros
  TStyle * style = getStyle("tdr");
//  style->SetTitleYOffset(1.6);
  style->cd();                      // Apply style 
  TCanvas* c1=new TCanvas("c1","c1",500,600);
  gPad->SetLogy();

  TLegend* l=new TLegend(.70,.50,.98,.78,"per SL");
  hnClusSL->SetXTitle("# rec");
  hnClusSL->SetLineWidth(2);
  hnClusSL->SetLineColor(kBlue);
  hnClusSL->SetFillColor(kBlue-10);
  hnClusSL->Draw();
  l->AddEntry(hnClusSL,"#clus","ls");
  hnSegSL->SetLineWidth(2);
  hnSegSL->SetLineColor(kRed);
  hnSegSL->Draw("same");
  l->AddEntry(hnSegSL,"#segs","ls");
  l->Draw();
  c1->Print("nClusSegSL.pdf");

  TCanvas* c2=new TCanvas("c2","c2",500,600);
  gPad->SetLogy();
  TLegend* l=new TLegend(.70,.50,.98,.78,"# hits");
  hClusNHits->SetXTitle("# hits");
  hClusNHits->SetLineWidth(2);
  hClusNHits->SetLineColor(kBlue);
  hClusNHits->SetFillColor(kBlue-10);
  hClusNHits->Draw();
  l->AddEntry(hClusNHits,"Clus","ls");
  hSegNHits->SetLineWidth(2);
  hSegNHits->SetLineColor(kRed);
  hSegNHits->Draw("same");
  l->AddEntry(hSegNHits,"Segs","ls");
  l->Draw();
  c2->Print("nHitsClusSeg.pdf");

  TCanvas* c3=new TCanvas("c3","c3",500,600);
  hClusVsSegPosSL->SetXTitle("x_{clus} (cm)");
  hClusVsSegPosSL->SetYTitle("x_{seg} (cm)");
  hClusVsSegPosSL->SetLineColor(kBlue);
  hClusVsSegPosSL->SetFillColor(kBlue);
  hClusVsSegPosSL->Draw("box");
  //hClusSegDistSL->GetXaxis()->SetRange(-10.,+10.);
  c3->Print("posClusSeg2D.pdf");

  TCanvas* c3b=new TCanvas("c3b","c3b",500,600);
  gPad->SetLogy();
  hClusSegDistSL->SetXTitle("#Delta{x}_{clus-seg} (cm)");
  hClusSegDistSL->SetLineColor(kBlue);
  hClusSegDistSL->SetFillColor(kBlue-10);
  hClusSegDistSL->Draw("same");
  c3b->Print("posClusSeg.pdf");

  TCanvas* c4=new TCanvas("c4","c4",500,600);
  hnClusVsSegs->SetXTitle("#_{segs}");
  hnClusVsSegs->SetYTitle("#_{clus}");
  hnClusVsSegs->SetLineColor(kBlue);
  hnClusVsSegs->SetFillColor(kBlue);
  hnClusVsSegs->SetMarkerColor(kBlue);
  hnClusVsSegs->Draw();
  c4->Print("nClusSeg.pdf");

}
