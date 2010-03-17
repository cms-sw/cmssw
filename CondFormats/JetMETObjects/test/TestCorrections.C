void TestCorrections()
{
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1);
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable(); 
  TRandom *rnd = new TRandom();
  rnd->SetSeed(0);
  ////////////// Construct the JetCorrectorParameters objects ///////////////////////
  string L3Tag = "../data/Summer09_7TeV_ReReco332_L3Absolute_AK5Calo.txt";
  string L2Tag = "../data/Summer09_7TeV_ReReco332_L2Relative_AK5Calo.txt";
  JetCorrectorParameters *L3JetPar = new JetCorrectorParameters(L3Tag);
  JetCorrectorParameters *L2JetPar = new JetCorrectorParameters(L2Tag);
  vector<JetCorrectorParameters> *vPar;
  vPar.push_back(L2JetPar);
  vPar.push_back(L3JetPar);
  ////////////// Construct a FactorizedJetCorrector object //////////////////////
  FactorizedJetCorrector *JetCorrector = new FactorizedJetCorrector(vPar) 
}
