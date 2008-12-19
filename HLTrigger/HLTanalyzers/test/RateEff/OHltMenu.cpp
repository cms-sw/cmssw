#include "OHltMenu.h"

using namespace std;

void OHltMenu::AddTrigger(TString trign, int presc, float eventS) {
  names.push_back(trign);
  prescales[trign] 	       	= presc;
  eventSizes[trign] 	       	= eventS;
}

void OHltMenu::AddL1forPreLoop(TString trign, int presc) {
  L1names.push_back(trign);
  L1prescales[trign] 	       	= presc;
}

void OHltMenu::print() {
  cout << "Menu - isL1Menu="<<isL1Menu << " - doL1preloop="<<doL1preloop<<  endl;
  cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;
  for (unsigned int i=0;i<names.size();i++) {
    cout<<names[i]<<" "<<prescales[names[i]]<<" "<<eventSizes[names[i]]<<" "<<endl;
  }
  cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;

  if (doL1preloop) {
    cout << endl << "L1 Menu - for L1preloop"<<  endl;
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;
    for (unsigned int i=0;i<L1names.size();i++) {
      cout<<L1names[i]<<" "<<L1prescales[L1names[i]]<<endl;
    }
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;
  }
  cout << endl;
}

void OHltMenu::SetMapL1SeedsOfStandardHLTPath() {
  typedef vector<TString> myvec;
  typedef pair< TString, vector<TString> > mypair;
  typedef map< TString, vector<TString> >  mymap;

  myvec vtmp;  

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Jet15", myvec(1,"L1_SingleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet30", myvec(1,"L1_SingleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet50", myvec(1,"L1_SingleJet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet80", myvec(1,"L1_SingleJet50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet110", myvec(1,"L1_SingleJet70")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet180", myvec(1,"L1_SingleJet70")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet250", myvec(1,"L1_SingleJet70")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_FwdJet20", myvec(1,"L1_IsoEG10_Jet15_ForJet10")));

  vtmp.clear(); vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet150", vtmp));
  vtmp.clear(); vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet125_Aco", vtmp));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleFwdJet50", myvec(1,"L1_SingleJet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve15", myvec(1,"L1_SingleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve30", myvec(1,"L1_SingleJet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve50", myvec(1,"L1_SingleJet50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve70", myvec(1,"L1_SingleJet70")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve130", myvec(1,"L1_SingleJet70")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve220", myvec(1,"L1_SingleJet70")));

  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet70"); vtmp.push_back("L1_TripleJet50");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_TripleJet85", vtmp));
    
  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet70");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet60", vtmp));
    
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet30", myvec(1,"L1_QuadJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_SumET120", myvec(1,"L1_ETT60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1MET20", myvec(1,"L1_ETM20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET25", myvec(1,"L1_ETM20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET35", myvec(1,"L1_ETM30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET50", myvec(1,"L1_ETM40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET65", myvec(1,"L1_ETM50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET75", myvec(1,"L1_ETM50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET35_HT350", myvec(1,"L1_HTT300")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet180_MET60", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet60_MET70_Aco", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet100_MET60_Aco", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet125_MET60", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleFwdJet40_MET60", myvec(1,"L1_ETM40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet60_MET60_Aco", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet50_MET70_Aco", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet40_MET70_Aco", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_TripleJet60_MET60", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet35_MET60", myvec(1,"L1_SingleJet150")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle15_L1I", myvec(1,"L1_SingleIsoEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle18_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle15_LW_L1I", myvec(1,"L1_SingleIsoEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_LooseIsoEle15_LW_L1R", myvec(1,"L1_SingleEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele10_SW_L1R", myvec(1,"L1_SingleEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele15_SW_L1R", myvec(1,"L1_SingleEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele15_LW_L1R", myvec(1,"L1_SingleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_EM80", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_EM200", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoEle10_L1I", myvec(1,"L1_DoubleIsoEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoEle12_L1R", myvec(1,"L1_DoubleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoEle10_LW_L1I", myvec(1,"L1_DoubleIsoEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoEle12_LW_L1R", myvec(1,"L1_DoubleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleEle5_SW_L1R", myvec(1,"L1_DoubleEG5")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleEle10_LW_OnlyPixelM_L1R", myvec(1,"L1_DoubleEG5")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleEle10_Z", myvec(1,"L1_DoubleIsoEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleEle6_Exclusive", myvec(1,"L1_ExclusiveDoubleIsoEG6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton30_L1I", myvec(1,"L1_SingleIsoEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton10_L1R", myvec(1,"L1_SingleEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton15_L1R", myvec(1,"L1_SingleEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton20_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton25_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton40_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon15_L1R", myvec(1,"L1_SingleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon25_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoPhoton20_L1I", myvec(1,"L1_DoubleIsoEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoPhoton20_L1R", myvec(1,"L1_DoubleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoublePhoton10_Exclusive", myvec(1,"L1_ExclusiveDoubleIsoEG6")));

  vtmp.clear(); vtmp.push_back("L1_SingleMu7"); vtmp.push_back("L1_DoubleMu3"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Mu", vtmp));
  
  vtmp.clear(); vtmp.push_back("L1_SingleMuOpen");
  vtmp.push_back("L1_SingleMu3"); vtmp.push_back("L1_SingleMu5");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1MuOpen", vtmp));
  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L2Mu9", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu9", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu11", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu13", myvec(1,"L1_SingleMu10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu15", myvec(1,"L1_SingleMu10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_NoTrackerIsoMu15", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu3", myvec(1,"L1_SingleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu5", myvec(1,"L1_SingleMu5")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu7", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu9", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu11", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu13", myvec(1,"L1_SingleMu10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu15", myvec(1,"L1_SingleMu10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu15_L1Mu7", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu15_Vtx2cm", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu15_Vtx2mm", myvec(1,"L1_SingleMu7")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoMu3", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_Vtx2cm", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_Vtx2mm", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_JPsi", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_Upsilon", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu7_Z", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_SameSign", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_Psi2S", myvec(1,"L1_DoubleMu3")));

  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet100");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_Jet180", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet100"); vtmp.push_back("L1_DoubleJet70");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_Jet120_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet100");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_DoubleJet120", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet100"); vtmp.push_back("L1_DoubleJet70");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_DoubleJet60_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet100");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_TripleJet70", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet100"); vtmp.push_back("L1_DoubleJet70");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_TripleJet40_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet100");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_QuadJet40", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet100"); vtmp.push_back("L1_DoubleJet70");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_QuadJet30_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_DoubleJet100");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_HT470", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet100"); vtmp.push_back("L1_DoubleJet70");
  vtmp.push_back("L1_TripleJet50"); vtmp.push_back("L1_QuadJet30"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_HT320_Relaxed", vtmp));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_DoubleJet120", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_DoubleJet60_Relaxed", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_TripleJet70", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_TripleJet40_Relaxed", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_QuadJet40", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_QuadJet30_Relaxed", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_HT370", myvec(1,"L1_HTT300")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_HT250_Relaxed", myvec(1,"L1_HTT200")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_BJPsi", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu4_BJPsi", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_TripleMu3_TauTo3Mu", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoTau_MET65_Trk20", myvec(1,"L1_SingleTauJet80")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoTau_MET35_Trk15_L1MET", myvec(1,"L1_TauJet30_ETM30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_LooseIsoTau_MET30", myvec(1,"L1_SingleTauJet80")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_LooseIsoTau_MET30_L1MET", myvec(1,"L1_TauJet30_ETM30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoTau_Trk3", myvec(1,"L1_DoubleTauJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleLooseIsoTau", myvec(1,"L1_DoubleTauJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle8_IsoMu7", myvec(1,"L1_Mu3_IsoEG5")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle10_Mu10_L1R", myvec(1,"L1_Mu3_EG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_IsoTau_Trk3", myvec(1,"L1_IsoEG10_TauJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle10_BTagIP_Jet35", myvec(1,"L1_IsoEG10_Jet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_Jet40", myvec(1,"L1_IsoEG10_Jet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_DoubleJet80", myvec(1,"L1_IsoEG10_Jet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoElec5_TripleJet30", myvec(1,"L1_EG5_TripleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_TripleJet60", myvec(1,"L1_IsoEG10_Jet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_QuadJet35", myvec(1,"L1_IsoEG10_Jet30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu14_IsoTau_Trk3", myvec(1,"L1_Mu5_TauJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu7_BTagIP_Jet35", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu7_BTagMu_Jet20", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu7_Jet40", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_NoL2IsoMu8_Jet40", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu14_Jet50", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu5_TripleJet30", myvec(1,"L1_Mu3_TripleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_Jet20_Calib", myvec(1,"L1_Mu5_Jet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_ZeroBias", myvec(1,"L1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBias", myvec(1,"L1_MinBias_HTT10")));

  vtmp.clear();
  vtmp.push_back("L1_SingleJetCountsHFTow"); vtmp.push_back("L1_DoubleJetCountsHFTow");
  vtmp.push_back("L1_SingleJetCountsHFRing0Sum3"); vtmp.push_back("L1_DoubleJetCountsHFRing0Sum3");
  vtmp.push_back("L1_SingleJetCountsHFRing0Sum6"); vtmp.push_back("L1_DoubleJetCountsHFRing0Sum6");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasHcal", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleEG2"); vtmp.push_back("L1_DoubleEG1");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasEcal", vtmp));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasPixel", myvec(1,"L1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasPixel_Trk5", myvec(1,"L1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHalo", myvec(1,"L1_SingleMuBeamHalo")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHaloOverlapRing1", myvec(1,"L1_SingleMuBeamHalo")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHaloOverlapRing2", myvec(1,"L1_SingleMuBeamHalo")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHaloRing2or3", myvec(1,"L1_SingleMuBeamHalo")));

  vtmp.clear();
  vtmp.push_back("L1_SingleJet30");
  vtmp.push_back("L1_SingleJet50"); vtmp.push_back("L1_SingleJet70");
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleTauJet40");
  vtmp.push_back("L1_SingleTauJet60"); vtmp.push_back("L1_SingleTauJet80");
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_IsoTrack", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_ZeroBias");
  vtmp.push_back("L1_SingleEG2"); vtmp.push_back("L1_DoubleEG1");
  vtmp.push_back("L1_SingleJetCountsHFTow"); vtmp.push_back("L1_DoubleJetCountsHFTow");
  vtmp.push_back("L1_SingleJetCountsHFRing0Sum3"); vtmp.push_back("L1_DoubleJetCountsHFRing0Sum3");
  vtmp.push_back("L1_SingleJetCountsHFRing0Sum6"); vtmp.push_back("L1_DoubleJetCountsHFRing0Sum6");
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_EcalPhiSym", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_ZeroBias");
  vtmp.push_back("L1_SingleEG2"); vtmp.push_back("L1_DoubleEG1");
  vtmp.push_back("L1_SingleJet30");vtmp.push_back("L1_SingleJet50");
  vtmp.push_back("L1_SingleJet70"); vtmp.push_back("L1_SingleJet100");
  vtmp.push_back("L1_SingleJet150"); vtmp.push_back("L1_SingleJet200");
  vtmp.push_back("L1_SingleIsoEG5"); vtmp.push_back("L1_SingleIsoEG8"); vtmp.push_back("L1_SingleIsoEG10");
  vtmp.push_back("L1_SingleIsoEG12"); vtmp.push_back("L1_SingleIsoEG15"); vtmp.push_back("L1_SingleIsoEG20");
  vtmp.push_back("L1_SingleEG10"); vtmp.push_back("L1_SingleEG12");
  vtmp.push_back("L1_SingleEG15"); vtmp.push_back("L1_SingleEG20"); vtmp.push_back("L1_SingleEG25");
  vtmp.push_back("L1_DoubleEG10"); vtmp.push_back("L1_DoubleEG15");
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_EcalPi0", vtmp));

    // For measuring L1 rates, also add L1 bits to the map!
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMuOpen", myvec(1,"L1_SingleMuOpen"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu3", myvec(1,"L1_SingleMu3"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu5", myvec(1,"L1_SingleMu5")));             
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu7", myvec(1,"L1_SingleMu7")));             
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu10", myvec(1,"L1_SingleMu10")));            
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMuBeamHalo", myvec(1,"L1_SingleMuBeamHalo"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleMu3", myvec(1,"L1_DoubleMu3")));             
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TripleMu3", myvec(1,"L1_TripleMu3")));             

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG5", myvec(1,"L1_SingleIsoEG5")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG8", myvec(1,"L1_SingleIsoEG8")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG10", myvec(1,"L1_SingleIsoEG10")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG12", myvec(1,"L1_SingleIsoEG12")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG15", myvec(1,"L1_SingleIsoEG15")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG25", myvec(1,"L1_SingleIsoEG25")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG20", myvec(1,"L1_SingleIsoEG20")));       
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleIsoEG8", myvec(1,"L1_DoubleIsoEG8")));         

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG2", myvec(1,"L1_SingleEG2")));            
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG5", myvec(1,"L1_SingleEG5")));             
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG8", myvec(1,"L1_SingleEG8")));             
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG10", myvec(1,"L1_SingleEG10")));            
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG12", myvec(1,"L1_SingleEG12")));           
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG15", myvec(1,"L1_SingleEG15")));          
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG20", myvec(1,"L1_SingleEG20")));          
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG25", myvec(1,"L1_SingleEG25")));          
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleEG1", myvec(1,"L1_DoubleEG1")));            
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleEG5", myvec(1,"L1_DoubleEG5")));             
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleEG10", myvec(1,"L1_DoubleEG10")));            

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet15", myvec(1,"L1_SingleJet15")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet30", myvec(1,"L1_SingleJet30"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet50", myvec(1,"L1_SingleJet50")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet70", myvec(1,"L1_SingleJet70")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet100", myvec(1,"L1_SingleJet100"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet150", myvec(1,"L1_SingleJet150"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet200", myvec(1,"L1_SingleJet200"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleJet70", myvec(1,"L1_DoubleJet70")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleJet100", myvec(1,"L1_DoubleJet100"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TripleJet50", myvec(1,"L1_TripleJet50")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_QuadJet15", myvec(1,"L1_QuadJet15")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_QuadJet30", myvec(1,"L1_QuadJet30")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_HTT200", myvec(1,"L1_HTT200")));         
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_HTT300", myvec(1,"L1_HTT300")));         

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM20", myvec(1,"L1_ETM20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM30", myvec(1,"L1_ETM30")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM40", myvec(1,"L1_ETM40")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM50", myvec(1,"L1_ETM50")));   

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETT60", myvec(1,"L1_ETT60")));  

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet30", myvec(1,"L1_SingleTauJet30")));                
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet40", myvec(1,"L1_SingleTauJet40")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet60", myvec(1,"L1_SingleTauJet60")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet80", myvec(1,"L1_SingleTauJet80")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleTauJet20", myvec(1,"L1_DoubleTauJet20")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleTauJet40", myvec(1,"L1_DoubleTauJet40")));    

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet15_ForJet10", myvec(1,"L1_IsoEG10_Jet15_ForJet10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ExclusiveDoubleIsoEG6", myvec(1,"L1_ExclusiveDoubleIsoEG6")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu5_Jet15", myvec(1,"L1_Mu5_Jet15")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet20", myvec(1,"L1_IsoEG10_Jet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet30", myvec(1,"L1_IsoEG10_Jet30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_IsoEG5", myvec(1,"L1_Mu3_IsoEG5")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_EG12", myvec(1,"L1_Mu3_EG12")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_TauJet20", myvec(1,"L1_IsoEG10_TauJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu5_TauJet20", myvec(1,"L1_Mu5_TauJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TauJet30_ETM30", myvec(1,"L1_TauJet30_ETM30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_EG5_TripleJet15", myvec(1,"L1_EG5_TripleJet15"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_TripleJet15", myvec(1,"L1_Mu3_TripleJet15"))); 

  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ZeroBias", myvec(1,"L1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_MinBias_HTT10", myvec(1,"L1_MinBias_HTT10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJetCountsHFTow", myvec(1,"L1_SingleJetCountsHFTow"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleJetCountsHFTow", myvec(1,"L1_DoubleJetCountsHFTow"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJetCountsHFRing0Sum3", myvec(1,"L1_SingleJetCountsHFRing0Sum3")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleJetCountsHFRing0Sum3", myvec(1,"L1_DoubleJetCountsHFRing0Sum3")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJetCountsHFRing0Sum6", myvec(1,"L1_SingleJetCountsHFRing0Sum6")));    

  /* New Taus */
  // This is openhlt and not standard hlt,
  // but the same mechanism can also be used here!
  // L1 prescales can be checked in the same way as 
  // for standard hlt in CheckOpenHlt(). 
  // Look for "New Taus" in OHltTreeOpen.cpp!
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet40"); vtmp.push_back("L1_SingleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20", vtmp));

  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet60"); vtmp.push_back("L1_SingleJet100");
  //vtmp.push_back("L1_SingleTauJet80"); vtmp.push_back("L1_SingleJet100");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk5", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet60"); vtmp.push_back("L1_SingleJet100");
  //vtmp.push_back("L1_SingleTauJet80"); vtmp.push_back("L1_SingleJet100");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk10", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet30"); vtmp.push_back("L1_DoubleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet30"); vtmp.push_back("L1_DoubleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_Trk5", vtmp));

  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet40"); vtmp.push_back("L1_SingleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_L2R", vtmp));

  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet60"); vtmp.push_back("L1_SingleJet100");
  //vtmp.push_back("L1_SingleTauJet80"); vtmp.push_back("L1_SingleJet100");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk5_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet60"); vtmp.push_back("L1_SingleJet100");
  //vtmp.push_back("L1_SingleTauJet80"); vtmp.push_back("L1_SingleJet100");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk10_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet30"); vtmp.push_back("L1_DoubleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet30"); vtmp.push_back("L1_DoubleJet70");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_Trk5_L2R", vtmp));

}
