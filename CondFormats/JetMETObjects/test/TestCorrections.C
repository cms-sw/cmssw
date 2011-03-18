void TestCorrections(double rawPt, double rawEta, double rawPhi, double rawE, double ZSPold, double JPTE, int NPV)
{
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  ////////////// Construct the JetCorrectorParameters objects ///////////////////////
  string L1Tag    = "Jec10V1_L1Offset_AK5JPT.txt"; 
  string L1JPTTag = "test_L1JPTOffset_AK5JPT.txt"; 
  JetCorrectorParameters *L1Par    = new JetCorrectorParameters(L1Tag);
  JetCorrectorParameters *L1JPTPar = new JetCorrectorParameters(L1JPTTag);
  vector<JetCorrectorParameters> vPar1,vPar2,vPar3;
  vPar1.push_back(*L1Par);
  vPar1.push_back(*L1JPTPar);
  vPar2.push_back(*L1JPTPar);
  vPar3.push_back(*L1Par);
  ////////////// Construct a FactorizedJetCorrector object //////////////////////
  FactorizedJetCorrector *JetCorrector1 = new FactorizedJetCorrector(vPar1);
  FactorizedJetCorrector *JetCorrector2 = new FactorizedJetCorrector(vPar2);
  FactorizedJetCorrector *JetCorrector3 = new FactorizedJetCorrector(vPar3);
  ////////////// Loop over jets //////////////////////
  TLorentzVector rawJetP4(0);
  rawJetP4.SetPtEtaPhiE(rawPt,rawEta,rawPhi,rawE);
  JetCorrector1->setJPTrawP4(rawJetP4); 
  JetCorrector1->setJPTrawZSP(ZSPold);
  JetCorrector1->setJetE(JPTE);
  JetCorrector1->setNPV(NPV);
  JetCorrector2->setJPTrawP4(rawJetP4); 
  JetCorrector2->setJPTrawZSP(ZSPold);
  JetCorrector2->setJetE(JPTE);
  JetCorrector3->setJPTrawP4(rawJetP4); 
  JetCorrector3->setNPV(NPV);
  vector<float> vcor1,vcor2,vcor3;
  vcor1 = JetCorrector1->getSubCorrections();
  vcor2 = JetCorrector2->getSubCorrections();
  vcor3 = JetCorrector3->getSubCorrections();
  cout<<"Correction applied to raw CaloJet after L1Offset = "<<vcor3[0]<<endl;
  cout<<"Correction applied to JPT jet after L1Offset = "<<vcor1[0]<<endl;
  cout<<"Correction applied to JPT jet after L1JPTOffset = "<<vcor1[1]<<endl;
  cout<<"Correction that would have been applied to JPT jet after L1JPTOffset without L1Offset = "<<vcor2[0]<<endl;
}
