// The FastSimulation must be recompiled with DEBUGCELLLINE enabled in FastSimulation/Utilities/interface/FamosDebug.h 
// before running testCellLine.cfg and finally this macro

{
  gStyle->SetOptStat(0);
  TFile * f = new TFile("Famos.root");
  h301->Draw();
  h302->SetMarkerColor(2);
  h302->Draw("same");
  h303->SetMarkerColor(4);
  h303->Draw("same");
  h304->SetMarkerColor(5);
  h304->Draw("same");
  h310->Draw("same");
}
