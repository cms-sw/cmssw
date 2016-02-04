void plotReso() {
  hHitResidualSeg->SetFillColor(kYellow-10);
  hHitResidualSeg->Draw();
  TLegend* l=new TLegend(.70,.75,.98,.98);
  l->AddEntry(hHitResidualSeg,"R+L","f");
  l->AddEntry(hHitResidualSegCellSX,"L","f");
  l->AddEntry(hHitResidualSegCellDX,"R","f");
  hHitResidualSegCellSX->SetLineColor(kGreen+2);
  hHitResidualSegCellSX->SetLineWidth(2);
  hHitResidualSegCellSX->Draw("same");
  hHitResidualSegCellDX->SetLineColor(kMagenta+1);
  hHitResidualSegCellDX->SetLineWidth(2);
  hHitResidualSegCellDX->Draw("same");
  l->Draw();
}
