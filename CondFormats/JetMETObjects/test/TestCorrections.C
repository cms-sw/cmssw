void TestCorrections(double eta, int NPV)
{
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  ////////////// Construct the JetCorrectorParameters objects ///////////////////////
  string L1Tag = "../data/Fall10_L1Offset_AK5PF.txt"; 
  JetCorrectorParameters *L1Par = new JetCorrectorParameters(L1Tag);
  vector<JetCorrectorParameters> vPar;
  vPar.push_back(*L1Par);
  ////////////// Construct a FactorizedJetCorrector object //////////////////////
  FactorizedJetCorrector *JetCorrector = new FactorizedJetCorrector(vPar);
  ////////////// Loop over jets //////////////////////
  double Emin = 10*TMath::CosH(eta);
  TH1F *h = new TH1F("histo","histo",100,Emin,7000);
  double r = pow(7000./Emin,1./100.);
  for(int ie=0;ie<100;ie++) {
    double energy = Emin*pow(r,ie);
    JetCorrector->setJetEta(eta);
    JetCorrector->setJetE(energy);
    JetCorrector->setNPV(NPV);
    h->SetBinContent(ie,JetCorrector->getCorrection());
  }
  TCanvas *can = new TCanvas();
  h->SetMinimum(0.85);
  h->SetMaximum(1.02);
  h->GetXaxis()->SetTitle("Jet Energy (GeV)");
  h->GetYaxis()->SetTitle("Correction Factor");
  h->Draw("][");
  TPaveText *pave = new TPaveText(0.4,0.3,0.7,0.6,"NDC");
  char name[1000];
  pave->AddText("L1Offset");
  sprintf(name,"|#eta| = %1.1f",eta);
  pave->AddText(name);
  sprintf(name,"NPV = %d",NPV);
  pave->AddText(name);
  sprintf(name,"p_{T,min} = %d GeV",10);
  pave->AddText(name);
  sprintf(name,"E_{min} = %d GeV",Emin);
  pave->AddText(name);
  pave->Draw();
}
