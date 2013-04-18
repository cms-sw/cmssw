void TestCorrections(double pt, double A, double Rho)
{
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  ////////////// Construct the JetCorrectorParameters objects ///////////////////////
  string L1Tag = "../data/Fall10_L1FastJet_AK5Calo.txt"; 
  JetCorrectorParameters *L1Par = new JetCorrectorParameters(L1Tag);
  vector<JetCorrectorParameters> vPar;
  vPar.push_back(*L1Par);
  ////////////// Construct a FactorizedJetCorrector object //////////////////////
  FactorizedJetCorrector *JetCorrector = new FactorizedJetCorrector(vPar);
  ////////////// Loop over jets //////////////////////
  JetCorrector->setJetEta(0.0); 
  JetCorrector->setJetPt(pt);
  JetCorrector->setJetA(A);
  JetCorrector->setRho(Rho);
  cout<<JetCorrector->getCorrection()<<endl;
}
