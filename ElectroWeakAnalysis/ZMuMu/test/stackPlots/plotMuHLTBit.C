{



TCut cut_zGolden("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0 &&  zGoldenDau1Chi2<10 && zGoldenDau2Chi2<10 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=11 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=11 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1TrackerMuonBit==1 && zGoldenDau2TrackerMuonBit==1 && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1   ");
 
TChain *  z = new TChain("Events");
// z.Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_ZmmPowhegSpring10HLTRedigi_100pb.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_132440_135802.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_135821-137731.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_138737-138751_promptreco_FF.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_138_919_939.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139020.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_096_103.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_195_239.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139347.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_356_360.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_362_365.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_368_370.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_372_375.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_399_411.root");
  z.Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_457_459.root");



  TCut eta1cut(" abs(zGoldenDau1Eta)>2.1 &&  abs(zGoldenDau1Eta) <2.4  ");
   TCut eta2cut("abs(zGoldenDau2Eta) < 2.4");

  z.Project("hltBit", "zGoldenDau1HLTBit", cut_zGolden + eta1cut + eta2cut);

 
  int N2 = hltBit.Integral() ;
  int N1 = hltBit.Integral(0,1) ;
  //  cout <<   "hltBit.Integral()" <<  N2 << endl;
  // cout <<   "hltBit.Integral(0,1)" <<  N1 << endl;


  TCut eta1cut("abs(zGoldenDau1Eta) < 2.4");
   TCut eta2cut("abs(zGoldenDau2Eta)> 2.1 &&  abs(zGoldenDau2Eta)  <2.4 ");

  z.Project("hltBit", "zGoldenDau2HLTBit", cut_zGolden + eta1cut + eta2cut);
 

  // cout <<   "hltBit.Integral()" <<  hltBit.Integral() << endl;
  // cout <<   "hltBit.Integral(0,1)" <<  hltBit.Integral(0,1) << endl;
  N2 += hltBit.Integral();
  N1 += hltBit.Integral(0,1);


  double eff= ((double) N2 - (double) (2 * N1)) / ((double) N2 - (double) ( N1)) ;
   cout <<   "eff " <<  eff << endl;

   cout << "N2 --> number of reco glb muons passing all cut ==" << N2 << endl;
   cout << "N2 - N1 --> number of reco glb muons passing the trigger path ==" << N2 - N1 << endl;
   cout << " eff = N2 - 2N1 / N2 -N1" << endl;
   cout << " N2 - 2N1 (successes) == " << N2 - 2 * N1<< endl;
   cout << " N2 - N1 (trials) == " << N2 -  N1<< endl;
}
