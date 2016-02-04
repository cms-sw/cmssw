void PlotDiffsAll (){

  TFile f1("test.root");
  
  TCanvas c1("c1","c1",10,10,800,1200);
  c1.Divide(3,4);

  c1.cd(1);
  theTree->Draw("dx");
 
  c1.cd(2);
  theTree->Draw("dy");

  c1.cd(3);
  theTree->Draw("dz");
  
  c1.cd(4);
  theTree->Draw("dtx");
 
  c1.cd(5);
  theTree->Draw("dty");

  c1.cd(6);
  theTree->Draw("dtz");

  c1.cd(7);
  theTree->Draw("dkx");

  c1.cd(8);
  theTree->Draw("dky");

  c1.cd(9);
  theTree->Draw("dkz");

  c1.cd(10);
  theTree->Draw("dnx");

  c1.cd(11);
  theTree->Draw("dny");
 
  c1.cd(12);
  theTree->Draw("dnz");
 
  c1.SaveAs("provaAll.eps");

  TCanvas c2("c2","c2",10,10,600,300);
  c2.Divide(2,1);

  c2.cd(1);
  theTree->Draw("sqrt(pow(dx,2)+pow(dy,2))");
 
  c2.cd(2);
  theTree->Draw("atan(dy/dx)");

  c2.SaveAs("provaRphi.eps");
}
