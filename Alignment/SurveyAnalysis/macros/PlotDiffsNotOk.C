void PlotDiffsNotOk (){

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
  theTree->Draw("dtx:Id","fabs(dtx)>0.04","BOX");
 
  c1.cd(5);
  theTree->Draw("dty:Id","fabs(dty)>0.04","BOX");

  c1.cd(6);
  theTree->Draw("dtz:Id","fabs(dtz)>0.006","BOX");

  c1.cd(7);
  theTree->Draw("dkx:Id","fabs(dkx)>0.04","BOX");

  c1.cd(8);
  theTree->Draw("dky:Id","fabs(dky)>0.04","BOX");

  c1.cd(9);
  theTree->Draw("dkz:Id","fabs(dkz)>0.02","BOX");

  c1.cd(10);
  theTree->Draw("dnx:Id","fabs(dnx)>0.04","BOX");

  c1.cd(11);
  theTree->Draw("dny:Id","fabs(dny)>0.04","BOX");
 
  c1.cd(12);
  theTree->Draw("dnz:Id","fabs(dnz)>0.04","BOX");
 
  c1.SaveAs("provaNotOk.eps"); 
}
