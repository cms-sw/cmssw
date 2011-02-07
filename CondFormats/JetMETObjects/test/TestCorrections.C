void TestCorrections(double pt, double eta)
{
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  ////////////// Construct the JetCorrectorParameters objects ///////////////////////
  string L3Tag = "../data/Spring10_L3Absolute_AK5Calo.txt";
  string L2Tag = "../data/Spring10_L2Relative_AK5Calo.txt";
  string L5Tag = "../data/Spring10_L5Flavor_AK5Calo.txt"; 
  string L7Tag = "../data/Spring10_L7Parton_AK5Calo.txt"; 
  ////////////// Construct a FactorizedJetCorrector object //////////////////////
  FactorizedJetCorrector *JetCorrector = new
  FactorizedJetCorrector("L2Relative:L3Absolute:L5Flavor:L7Parton",L2Tag+":"+L3Tag+":"+L5Tag+":"+L7Tag,"L5Flavor:qJ&L7Parton:qJ");
  JetCorrector->setJetEta(eta);
  JetCorrector->setJetPt(pt);
  vector<float> v;
  v = JetCorrector->getSubCorrections();
  for(int i=0;i<4;i++)
    cout<<v[i]<<endl;
}
