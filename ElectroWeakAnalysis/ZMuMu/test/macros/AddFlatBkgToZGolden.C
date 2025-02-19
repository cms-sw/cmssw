{
gROOT->SetStyle("Plain");
TFile * root_file = new TFile("Analysis_7TeV.root", "update");

TH1 * histoZMuMu = root_file->Get("goodZToMuMuPlots/zMass");
TH1 * histoZMuMu1HLT = root_file->Get("goodZToMuMu1HLTPlots/zMass");
TH1 * histoZMuMu2HLT = root_file->Get("goodZToMuMu2HLTPlots/zMass");
double npercent = 1.;

//histoZMuMu->Draw();
double integ = (double) histoZMuMu->Integral();
cout << "integ: " << integ << endl;
for(int i = 1; i <= histoZMuMu->GetNbinsX(); ++i) {
  double cont = histoZMuMu->GetBinContent(i) ; 
  //cout << "cont: " << cont << endl;
  // adding npercent flat content....
  double new_cont= cont  + ( 0.01 * npercent  * integ /  histoZMuMu->GetNbinsX());
  //cout << "new_cont: " << new_cont << endl;
  histoZMuMu->SetBinContent(i, new_cont);
  histoZMuMu->SetBinError(i, sqrt(new_cont));
}


integ = (double) histoZMuMu1HLT->Integral();
cout << "integ: " << integ << endl;
for(int i = 1; i <= histoZMuMu1HLT->GetNbinsX(); ++i) {
  double cont = histoZMuMu1HLT->GetBinContent(i) ; 
  //cout << "cont: " << cont << endl;
  // adding npercent flat content....
  double new_cont= cont  + ( 0.01 * npercent  * integ /  histoZMuMu1HLT->GetNbinsX());
  //cout << "new_cont: " << new_cont << endl;
  histoZMuMu1HLT->SetBinContent(i, new_cont);
  histoZMuMu1HLT->SetBinError(i, sqrt(new_cont));
}


integ = (double) histoZMuMu2HLT->Integral();
cout << "integ: " << integ << endl;
for(int i = 1; i <= histoZMuMu2HLT->GetNbinsX(); ++i) {
  double cont = histoZMuMu2HLT->GetBinContent(i) ; 
  //cout << "cont: " << cont << endl;
  // adding npercent flat content....
  double new_cont= cont  + ( 0.01 * npercent  * integ /  histoZMuMu2HLT->GetNbinsX());
  //cout << "new_cont: " << new_cont << endl;
  histoZMuMu2HLT->SetBinContent(i, new_cont);
  histoZMuMu2HLT->SetBinError(i, sqrt(new_cont));
}



root_file->cd("goodZToMuMuPlots");
histoZMuMu->Write();

root_file->cd("goodZToMuMu1HLTPlots");
histoZMuMu1HLT->Write();

root_file->cd("goodZToMuMu2HLTPlots");
histoZMuMu2HLT->Write();
double  new_integ = (double) histoZMuMu->Integral();
cout << "new integ: " << new_integ << endl;
new_integ = (double) histoZMuMu1HLT->Integral();
cout << "new integ: " << new_integ << endl;
new_integ = (double) histoZMuMu2HLT->Integral();
cout << "new integ: " << new_integ << endl;
}
