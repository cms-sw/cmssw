/*************************/
/*                       */
/* author: Pasquale Noli */
/* INFN Naples           */
/* macro to save the eps */ 
/* of pulls              */
/*                       */
/*************************/

{
  TFile *f = TFile::Open("fitResult.root"); 
  TH1D frameYield("frameYield", "Yield", 100, -10, 10);
  TH1D frameTrk("frameTrk", "track eff.", 100, -10, 10);
  TH1D frameSa("frameSa", "stand-alone eff.", 100, -10, 10);
  TH1D frameIso("frameIso", "isolation eff.", 100, -10, 10);
  TH1D frameHlt("frameHlt", "HLT eff.", 100, -10, 10);
  TH1D frameErrYield("frameErrYield", "Err Yield", 100, 0, 100);
  TH1D frameErrTrk("frameErrTrk", "Err track eff.", 100, 0, .01);
  TH1D frameErrSa("frameErrSa", "Err stand-alone eff.", 100, 0, .01);
  TH1D frameErrIso("frameErrIso", "Err isolation eff.", 100, 0, .01);
  TH1D frameErrHlt("frameErrHlt", "Err HLT eff.", 100, 0, .01);
  TH1D frameChi2("frameChi2", "chi2", 100, 0, 10);

  tree->Project("frameYield","(Y-Y_true)/dY", "abs((Y - Y_true)/dY)>5.e-3");
  tree->Project("frameTrk","(Tk-Tk_true)/dTk", "abs((Tk-Tk_true)/dTk)>5.e-3");
  tree->Project("frameSa","(Sa-Sa_true)/dSa", "abs((Sa-Sa_true)/dSa)>5.e-3"); 
  tree->Project("frameIso", "(Iso-Iso_true)/dIso", "abs((Iso-Iso_true)/dIso)>5.e-3");
  tree->Project("frameHlt", "(Hlt-Hlt_true)/dHlt", "abs((Hlt-Hlt_true)/dHlt)>5.e-3");
  
  tree->Project("frameErrYield","dY", "abs(dY)>5.e-5");
  tree->Project("frameErrTrk","dTk", "abs(dTk)>5.e-5");
  tree->Project("frameErrSa","dSa", "abs(dSa)>5.e-5"); 
  tree->Project("frameErrIso", "dIso", "abs(dIso)>5.e-5");
  tree->Project("frameErrHlt", "dHlt", "abs(dHlt)>5.e-5");
  tree->Project("frameChi2", "chi2", "abs(chi2)>5.e-3");
  


  frameYield.Fit("gaus");  
  frameTrk.Fit("gaus");
  frameSa.Fit("gaus");
  frameIso.Fit("gaus");
  frameHlt.Fit("gaus");
  TCanvas *c1 = new TCanvas("c1","pulls",10,10,900,900);
  gStyle->SetOptStat(1111111);
  gStyle->SetStatFontSize(0.04);
  gStyle->SetOptFit(kTRUE);
  gStyle->SetFitFormat("5.3g");
  c1->Divide (3,2);
  
  c1->cd(1);
  frameYield.Draw();

  c1->cd(2);
  frameTrk.Draw();

  c1->cd(3);

  frameSa.Draw();
  c1->cd(4);
  frameIso.Draw();
  c1->cd(5);
  frameHlt.Draw();
  c1->Draw();
  c1->SaveAs("pulls.eps");

  TCanvas *c2 = new TCanvas("c2","err",10,10,900,900);
  c2->Divide (3,2);
  c2->cd(1);
  frameErrYield.Draw();

  c2->cd(2);
  frameErrTrk.Draw();

  c2->cd(3);

  frameErrSa.Draw();
  c2->cd(4);
  frameErrIso.Draw();
  c2->cd(5);
  frameErrHlt.Draw();

 
  c2->cd(6);
  frameChi2.Draw();
  c2->Draw();
  c2->SaveAs("Err.eps");
 



}

