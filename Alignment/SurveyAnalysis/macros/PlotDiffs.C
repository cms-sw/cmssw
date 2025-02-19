void PlotDiffs (){

  TFile f1("test1.root");
  f1.cd();
  TTree * tree1 = (TTree *) gROOT->FindObject("theTree");
  TFile f2("test2.root");
  f2.cd();
  TTree * tree2 = (TTree *) gROOT->FindObject("theTree");
  TFile f3("test3.root");
  f3.cd();
  TTree * tree3 = (TTree *) gROOT->FindObject("theTree");
  TFile f4("test4.root");
  f4.cd();
  TTree * tree4 = (TTree *) gROOT->FindObject("theTree");

  TCanvas c1("c1","c1",10,10,800,800);
  c1.Divide(3,3);

  c1.cd(1);
  tree4->Draw("dtx");
  tree2->Draw("dtx","","pesame");
  // tree3->Draw("dtx","","e1same");
  // tree4->Draw("dtx","","e2same");

  c1.cd(2);
  tree4->Draw("dty");
  tree2->Draw("dty","","pesame");
  // tree3->Draw("dty","","e1same");
  // tree4->Draw("dty","","e2same");

  c1.cd(3);
  tree4->Draw("dtz");
  tree2->Draw("dtz","","pesame");
  // tree3->Draw("dtz","","e1same");
  // tree4->Draw("dtz","","e2same");  

  c1.cd(4);
  tree4->Draw("dkx");
  tree2->Draw("dkx","","pesame");
  // tree3->Draw("dkx","","e1same");
  // tree4->Draw("dkx","","e2same");  

  c1.cd(5);
  tree4->Draw("dky");
  tree2->Draw("dky","","pesame");
  // tree3->Draw("dky","","e1same");
  // tree4->Draw("dky","","e2same");

  c1.cd(6);
  tree4->Draw("dkz");
  tree2->Draw("dkz","","pesame");
  // tree3->Draw("dkz","","e1same");
  // tree4->Draw("dkz","","e2same");

  c1.cd(7);
  tree4->Draw("dnx");
  tree2->Draw("dnx","","pesame");
  // tree3->Draw("dnx","","e1same");
  // tree4->Draw("dnx","","e2same");

  c1.cd(8);
  tree4->Draw("dny");
  tree2->Draw("dny","","pesame");
  // tree3->Draw("dny","","e1same");
  // tree4->Draw("dny","","e2same");

  c1.cd(9);
  tree4->Draw("dnz");
  tree2->Draw("dnz","","pesame");
  // tree3->Draw("dnz","","e1same");
  // tree4->Draw("dnz","","e2same");

  c1.SaveAs("prova.eps");
}
