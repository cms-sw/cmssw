void TestCorrections(double rawPt, double rawEta, double rawPhi, double rawE, double JPTE, int NPV)
{
  gROOT->ProcessLine("#include <vector>");
  gSystem->Load("libFWCoreFWLite.so");
  FWLiteEnabler::enable();
  ////////////// Construct the JetCorrectorParameters objects ///////////////////////
  string L1Tag    = "../data/Jec11V0_L1Offset_AK5JPT.txt"; 
  string L1JPTTag = "../data/Jec11V0_L1JPTOffset_AK5JPT.txt"; 
  JetCorrectorParameters *L1Par    = new JetCorrectorParameters(L1Tag);
  JetCorrectorParameters *L1JPTPar = new JetCorrectorParameters(L1JPTTag);
  vector<JetCorrectorParameters> vPar;
  vPar.push_back(*L1Par);
  vPar.push_back(*L1JPTPar);
  ////////////// Construct a FactorizedJetCorrector object //////////////////////
  FactorizedJetCorrector *JetCorrector = new FactorizedJetCorrector(vPar);
  ////////////// Loop over jets //////////////////////
  TLorentzVector rawJetP4(0);
  rawJetP4.SetPtEtaPhiE(rawPt,rawEta,rawPhi,rawE);
  JetCorrector->setJPTrawP4(rawJetP4); 
  JetCorrector->setJetE(JPTE);
  JetCorrector->setNPV(NPV);
  vector<float> vcor;
  vcor = JetCorrector->getSubCorrections();
  cout<<"Correction applied to JPT jet after L1Offset = "<<vcor[0]<<endl;
  cout<<"Correction applied to JPT jet after L1JPTOffset = "<<vcor[1]<<endl;
}
