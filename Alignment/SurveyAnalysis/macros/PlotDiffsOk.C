void PlotDiffsOk (){

  TFile f1("test.root");
  
  TCanvas c1("c1","c1",10,10,800,1200);
  c1.Divide(3,4);
  TCanvas c2("c2","c2",10,10,800,1200);
  c2.Divide(3,4);

  c1.cd(1);
  theTree->Draw("dx");
 
  c1.cd(2);
  theTree->Draw("dy");
  
  c1.cd(3);
  theTree->Draw("dz");
  
  c1.cd(4);
  theTree->Draw("dtx","fabs(dtx)<0.04");
 
  c1.cd(5);
  theTree->Draw("dty","fabs(dty)<0.04");

  c1.cd(6);
  theTree->Draw("dtz","fabs(dtz)<0.006");

  c1.cd(7);
  theTree->Draw("dkx","fabs(dkx)<0.04");

  c1.cd(8);
  theTree->Draw("dky","fabs(dky)<0.04");

  c1.cd(9);
  theTree->Draw("dkz","fabs(dkz)<0.02");

  c1.cd(10);
  theTree->Draw("dnx","fabs(dnx)<0.04");

  c1.cd(11);
  theTree->Draw("dny","fabs(dny)<0.04");
 
  c1.cd(12);
  theTree->Draw("dnz","fabs(dnz)<0.04");

  c1.SaveAs("provaOk.eps");

  ////
  
  c2.cd(1);
  theTree->Draw("dx:Id");

  c2.cd(2);
  theTree->Draw("dy:Id");
  
  c2.cd(3);
  theTree->Draw("dz:Id");
  
  c2.cd(4);
  theTree->Draw("dtx:Id","fabs(dtx)<0.04");
 
  c2.cd(5);
  theTree->Draw("dty:Id","fabs(dty)<0.04");

  c2.cd(6);
  theTree->Draw("dtz:Id","fabs(dtz)<0.006");

  c2.cd(7);
  theTree->Draw("dkx:Id","fabs(dkx)<0.04");

  c2.cd(8);
  theTree->Draw("dky:Id","fabs(dky)<0.04");

  c2.cd(9);
  theTree->Draw("dkz:Id","fabs(dkz)<0.02");

  c2.cd(10);
  theTree->Draw("dnx:Id","fabs(dnx)<0.04");

  c2.cd(11);
  theTree->Draw("dny:Id","fabs(dny)<0.04");
 
  c2.cd(12);
  theTree->Draw("dnz:Id","fabs(dnz)<0.04");

  c2.SaveAs("provaOkScatt.eps");

  TCanvas c3("c3","c3",10,10,900,300);
  c3.Divide(3,1);
  c3.cd(1);
  theTree->Draw("errx");
 
  c3.cd(2);
  theTree->Draw("erry");
  
  c3.cd(3);
  theTree->Draw("errz");
  c3.SaveAs("provaErr.eps");
 
  TCanvas c4("c4","c4",10,10,900,300);
  c4.Divide(3,1);
  c4.cd(1);
  theTree->Draw("errx:Id");
 
  c4.cd(2);
  theTree->Draw("erry:Id");
  
  c4.cd(3);
  theTree->Draw("errz:Id");
  c4.SaveAs("provaErrScatt.eps");
}
