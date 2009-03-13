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

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Jet15", myvec(1,"L1_SingleJet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet30", myvec(1,"L1_SingleJet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet50", myvec(1,"L1_SingleJet10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet80", myvec(1,"L1_SingleJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet110", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet180", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet250", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_FwdJet20", myvec(1,"L1_IsoEG10_Jet15_ForJet10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Jet6U", myvec(1,"L1_SingleJet6"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet15U", myvec(1,"L1_SingleJet6")));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet30U", myvec(1,"L1_SingleJet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet50U", myvec(1,"L1_SingleJet10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet80U", myvec(1,"L1_SingleJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet110U", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet180U", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet250U", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_FwdJet20U", myvec(1,"L1_IsoEG10_Jet15_ForJet10")));

  vtmp.clear(); vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet150", vtmp));
  vtmp.clear(); vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet125_Aco", vtmp));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleFwdJet50", myvec(1,"L1_SingleJet10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve15U", myvec(1,"L1_SingleJet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve30U", myvec(1,"L1_SingleJet10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve50", myvec(1,"L1_SingleJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve70", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve130", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DiJetAve220", myvec(1,"L1_SingleJet40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet15U", myvec(1, "L1_QuadJet6")));

  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40"); vtmp.push_back("L1_TripleJet30");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_TripleJet85", vtmp));
    
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet60", vtmp));
    
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet30", myvec(1,"L1_QuadJet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_SumET120", myvec(1,"L1_ETT60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1MET20", myvec(1,"L1_ETM20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET25", myvec(1,"L1_ETM20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET35", myvec(1,"L1_ETM30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET50", myvec(1,"L1_ETM40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET65", myvec(1,"L1_ETM50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET75", myvec(1,"L1_ETM50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET100", myvec(1,"L1_ETM80"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MET35_HT350", myvec(1,"L1_HTT300")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet180_MET60", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet60_MET70_Aco", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Jet100_MET60_Aco", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet125_MET60", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleFwdJet40_MET60", myvec(1,"L1_ETM40")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet60_MET60_Aco", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet50_MET70_Aco", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleJet40_MET70_Aco", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_TripleJet60_MET60", myvec(1,"L1_SingleJet60")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_QuadJet35_MET60", myvec(1,"L1_SingleJet60")));
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
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton30_L1I", myvec(1,"L1_SingleIsoEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton10_L1R", myvec(1,"L1_SingleEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton15_L1R", myvec(1,"L1_SingleEG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton20_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton25_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoPhoton40_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon15_L1R", myvec(1,"L1_SingleEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon15_LooseEcalIso_L1R", myvec(1,"L1_SingleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon25_L1R", myvec(1,"L1_SingleEG15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoPhoton20_L1I", myvec(1,"L1_DoubleIsoEG8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoPhoton20_L1R", myvec(1,"L1_DoubleEG10")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1SingleEG5", myvec(1,"L1_SingleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1SingleEG8", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele10_LW_L1R", myvec(1,"L1_SingleEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele15_LW_L1R", myvec(1,"L1_SingleEG8")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1DoubleEG5", myvec(1,"L1_DoubleEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele15_SC10_LW_L1R", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele20_LW_L1R", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Ele25_LW_L1R", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon10_L1R", myvec(1,"L1_SingleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon15_TrackIso_L1R", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon20_L1R", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Photon30_L1R", myvec(1,"L1_SingleEG8")));   

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
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu0", myvec(1,"L1_DoubleMuOpen")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1DoubleMuOpen", myvec(1,"L1_DoubleMuOpen")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu3", myvec(1,"L1_SingleMu3")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L2Mu11", myvec(1,"L1_SingleMu7")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Mu20", myvec(1,"L1_SingleMu20")));    
 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_Jet50U_ST", myvec(1,"L1_SingleJet30")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_Jet10U_Calib", myvec(1,"L1_Mu3_Jet6")));     



  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet60");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_Jet180", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_Jet120_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet60");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_DoubleJet120", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_DoubleJet60_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet60");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_TripleJet70", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_TripleJet40_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet60");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_QuadJet40", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_QuadJet30_Relaxed", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet60");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_HT470", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleJet60"); vtmp.push_back("L1_DoubleJet40");
  vtmp.push_back("L1_TripleJet30"); vtmp.push_back("L1_QuadJet20"); vtmp.push_back("L1_HTT300");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagIP_HT320_Relaxed", vtmp));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_DoubleJet120", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_DoubleJet60_Relaxed", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_TripleJet70", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_TripleJet40_Relaxed", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_QuadJet40", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_QuadJet30_Relaxed", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_HT370", myvec(1,"L1_HTT300")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_HT250_Relaxed", myvec(1,"L1_HTT200")));

  vtmp.clear(); 
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet30"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_SingleLooseIsoTau20", vtmp)); 

  vtmp.clear();
  vtmp.push_back("L1_DoubleJet30"); vtmp.push_back("L1_DoubleTauJet14");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleLooseIsoTau15", vtmp));

  vtmp.clear(); 
  vtmp.push_back("L1_SingleMu14"); vtmp.push_back("L1_SingleEG10"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Mu14_L1SingleEG10", vtmp));

  vtmp.clear();  
  vtmp.push_back("L1_SingleMu14"); vtmp.push_back("L1_SingleJet6");  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Mu14_L1SingleJet6U", vtmp)); 

  vtmp.clear();  
  vtmp.push_back("L1_SingleMu14"); vtmp.push_back("L1_ETM30");  
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_L1Mu14_L1ETM30", vtmp)); 

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_DoubleJet120", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_DoubleJet60_Relaxed", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_TripleJet70", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_TripleJet40_Relaxed", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_QuadJet40", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_QuadJet30_Relaxed", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_HT370", myvec(1,"L1_HTT300")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_HT250_Relaxed", myvec(1,"L1_HTT200")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu3_BJPsi", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleMu4_BJPsi", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_TripleMu3_TauTo3Mu", myvec(1,"L1_DoubleMu3")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoTau_MET65_Trk20", myvec(1,"L1_SingleTauJet50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoTau_MET35_Trk15_L1MET", myvec(1,"L1_TauJet10_ETM30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_LooseIsoTau_MET30", myvec(1,"L1_SingleTauJet50")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_LooseIsoTau_MET30_L1MET", myvec(1,"L1_TauJet10_ETM30")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleIsoTau_Trk3", myvec(1,"L1_DoubleTauJet20")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_DoubleLooseIsoTau", myvec(1,"L1_DoubleTauJet8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle8_IsoMu7", myvec(1,"L1_Mu3_IsoEG5")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle10_Mu10_L1R", myvec(1,"L1_Mu3_EG12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_IsoTau_Trk3", myvec(1,"L1_IsoEG10_TauJet8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle10_BTagIP_Jet35", myvec(1,"L1_IsoEG10_Jet8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_Jet40", myvec(1,"L1_IsoEG10_Jet12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_DoubleJet80", myvec(1,"L1_IsoEG10_Jet12")));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoElec5_TripleJet30", myvec(1,"L1_EG5_TripleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_TripleJet60", myvec(1,"L1_IsoEG10_Jet12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoEle12_QuadJet35", myvec(1,"L1_IsoEG10_Jet12")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu14_IsoTau_Trk3", myvec(1,"L1_Mu5_TauJet8")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu7_BTagIP_Jet35", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu7_BTagMu_Jet20", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_IsoMu7_Jet40", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_NoL2IsoMu8_Jet40", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu14_Jet50", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_Mu5_TripleJet30", myvec(1,"L1_Mu3_TripleJet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_BTagMu_Jet20_Calib", myvec(1,"L1_Mu5_Jet6")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_ZeroBias", myvec(1,"OpenL1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBias", myvec(1,"L1_MinBias_HTT10")));

   
  vtmp.clear(); 
  vtmp.push_back("L1_SingleEG1"); vtmp.push_back("L1_DoubleEG1"); vtmp.push_back("L1_SingleEG2"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_HcalPhiSym", vtmp)); 
 
  vtmp.clear();
  vtmp.push_back("L1_SingleHfBitCountsRing1_1"); vtmp.push_back("L1_DoubleHfBitCountsRing1_P1N1");
  vtmp.push_back("L1_SingleHfRingEtSumsRing1_4"); vtmp.push_back("L1_DoubleHfRingEtSumsRing1_P4N4");
  vtmp.push_back("L1_SingleHfRingEtSumsRing2_4"); vtmp.push_back("L1_DoubleHfRingEtSumsRing2_P4N4");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasHcal", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleEG1"); vtmp.push_back("L1_DoubleEG1");
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasEcal", vtmp));

  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasPixel", myvec(1,"OpenL1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_MinBiasPixel_Trk5", myvec(1,"OpenL1_ZeroBias")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHalo", myvec(1,"L1_SingleMuBeamHalo")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHaloOverlapRing1", myvec(1,"L1_SingleMuBeamHalo")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHaloOverlapRing2", myvec(1,"L1_SingleMuBeamHalo")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("HLT_CSCBeamHaloRing2or3", myvec(1,"L1_SingleMuBeamHalo")));

  vtmp.clear();
  vtmp.push_back("L1_SingleJet10");
  vtmp.push_back("L1_SingleJet20"); vtmp.push_back("L1_SingleJet40");
  vtmp.push_back("L1_SingleTauJet10"); vtmp.push_back("L1_SingleTauJet20");
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleTauJet50");
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_IsoTrack", vtmp));
  
  vtmp.clear();
  vtmp.push_back("OpenL1_ZeroBias");
  vtmp.push_back("L1_SingleEG1"); vtmp.push_back("L1_DoubleEG1");
  vtmp.push_back("L1_SingleHfBitCountsRing1_1"); vtmp.push_back("L1_DoubleHfBitCountsRing1_P1N1");
  vtmp.push_back("L1_SingleHfRingEtSumsRing1_4"); vtmp.push_back("L1_DoubleHfRingEtSumsRing1_P4N4");
  vtmp.push_back("L1_SingleHfRingEtSumsRing2_4"); vtmp.push_back("L1_DoubleHfRingEtSumsRing2_P4N4");
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_EcalPhiSym", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleIsoEG5");   vtmp.push_back("L1_SingleIsoEG8");  
  vtmp.push_back("L1_SingleIsoEG10"); vtmp.push_back("L1_SingleIsoEG12");
  vtmp.push_back("L1_SingleIsoEG15"); vtmp.push_back("L1_SingleEG1");  
  vtmp.push_back("L1_SingleEG2"); vtmp.push_back("L1_SingleEG5"); vtmp.push_back("L1_SingleEG8");   
  vtmp.push_back("L1_SingleEG10"); vtmp.push_back("L1_SingleEG12");
  vtmp.push_back("L1_SingleEG15"); vtmp.push_back("L1_SingleEG20"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("AlCa_EcalPi0", vtmp));

    // For measuring L1 rates, also add L1 bits to the map!
  /* New L1s from the L1Menu_startup2_v2 */
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleEG10", myvec(1,"L1_DoubleEG10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleEG1", myvec(1,"L1_DoubleEG1")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleEG5", myvec(1,"L1_DoubleEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleForJet20", myvec(1,"L1_DoubleForJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleHfBitCountsRing1_P1N1", myvec(1,"L1_DoubleHfBitCountsRing1_P1N1")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleHfBitCountsRing2_P1N1", myvec(1,"L1_DoubleHfBitCountsRing2_P1N1")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleHfRingEtSumsRing1_P200N200", myvec(1,"L1_DoubleHfRingEtSumsRing1_P200N200")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleHfRingEtSumsRing1_P4N4", myvec(1,"L1_DoubleHfRingEtSumsRing1_P4N4")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleHfRingEtSumsRing2_P200N200", myvec(1,"L1_DoubleHfRingEtSumsRing2_P200N200")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleHfRingEtSumsRing2_P4N4", myvec(1,"L1_DoubleHfRingEtSumsRing2_P4N4")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleIsoEG05_TopBottom", myvec(1,"L1_DoubleIsoEG05_TopBottom")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleIsoEG05_TopBottomCen", myvec(1,"L1_DoubleIsoEG05_TopBottomCen")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleIsoEG10", myvec(1,"L1_DoubleIsoEG10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleIsoEG8", myvec(1,"L1_DoubleIsoEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleJet40", myvec(1,"L1_DoubleJet40")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleJet60", myvec(1,"L1_DoubleJet60")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleMu3", myvec(1,"L1_DoubleMu3")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleMuOpen", myvec(1,"L1_DoubleMuOpen")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleMuTopBottom", myvec(1,"L1_DoubleMuTopBottom")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleNoIsoEG05_TopBottom", myvec(1,"L1_DoubleNoIsoEG05_TopBottom")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleNoIsoEG05_TopBottomCen", myvec(1,"L1_DoubleNoIsoEG05_TopBottomCen")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleTauJet20", myvec(1,"L1_DoubleTauJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_DoubleTauJet8", myvec(1,"L1_DoubleTauJet8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_EG12_Jet40", myvec(1,"L1_EG12_Jet40")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_EG5_TripleJet6", myvec(1,"L1_EG5_TripleJet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM20", myvec(1,"L1_ETM20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM30", myvec(1,"L1_ETM30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM40", myvec(1,"L1_ETM40")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETM50", myvec(1,"L1_ETM50")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_ETT60", myvec(1,"L1_ETT60")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_HTT100", myvec(1,"L1_HTT100")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_HTT200", myvec(1,"L1_HTT200")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_HTT300", myvec(1,"L1_HTT300")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet12", myvec(1,"L1_IsoEG10_Jet12")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet6", myvec(1,"L1_IsoEG10_Jet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet6_ForJet6", myvec(1,"L1_IsoEG10_Jet6_ForJet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_Jet8", myvec(1,"L1_IsoEG10_Jet8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_IsoEG10_TauJet8", myvec(1,"L1_IsoEG10_TauJet8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_MinBias_ETT10", myvec(1,"L1_MinBias_ETT10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_MinBias_HTT10", myvec(1,"L1_MinBias_HTT10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_EG12", myvec(1,"L1_Mu3_EG12")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_IsoEG5", myvec(1,"L1_Mu3_IsoEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_Jet6", myvec(1,"L1_Mu3_Jet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu3_TripleJet6", myvec(1,"L1_Mu3_TripleJet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu5_IsoEG10", myvec(1,"L1_Mu5_IsoEG10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu5_Jet6", myvec(1,"L1_Mu5_Jet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_Mu5_TauJet8", myvec(1,"L1_Mu5_TauJet8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_QuadJet20", myvec(1,"L1_QuadJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_QuadJet6", myvec(1,"L1_QuadJet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG1", myvec(1,"L1_SingleEG1")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG10", myvec(1,"L1_SingleEG10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG12", myvec(1,"L1_SingleEG12")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG15", myvec(1,"L1_SingleEG15")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG20", myvec(1,"L1_SingleEG20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG5", myvec(1,"L1_SingleEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG5_Endcap", myvec(1,"L1_SingleEG5_Endcap")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleEG8", myvec(1,"L1_SingleEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleForJet10", myvec(1,"L1_SingleForJet10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleForJet6", myvec(1,"L1_SingleForJet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleHfBitCountsRing1_1", myvec(1,"L1_SingleHfBitCountsRing1_1")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleHfBitCountsRing2_1", myvec(1,"L1_SingleHfBitCountsRing2_1")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleHfRingEtSumsRing1_200", myvec(1,"L1_SingleHfRingEtSumsRing1_200")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleHfRingEtSumsRing1_4", myvec(1,"L1_SingleHfRingEtSumsRing1_4")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleHfRingEtSumsRing2_200", myvec(1,"L1_SingleHfRingEtSumsRing2_200")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleHfRingEtSumsRing2_4", myvec(1,"L1_SingleHfRingEtSumsRing2_4")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG10", myvec(1,"L1_SingleIsoEG10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG12", myvec(1,"L1_SingleIsoEG12")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG15", myvec(1,"L1_SingleIsoEG15")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG5", myvec(1,"L1_SingleIsoEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG5_Endcap", myvec(1,"L1_SingleIsoEG5_Endcap")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleIsoEG8", myvec(1,"L1_SingleIsoEG8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet10", myvec(1,"L1_SingleJet10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet10_Barrel", myvec(1,"L1_SingleJet10_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet10_Central", myvec(1,"L1_SingleJet10_Central")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet10_Endcap", myvec(1,"L1_SingleJet10_Endcap")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet20", myvec(1,"L1_SingleJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet20_Barrel", myvec(1,"L1_SingleJet20_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet30", myvec(1,"L1_SingleJet30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet30_Barrel", myvec(1,"L1_SingleJet30_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet40", myvec(1,"L1_SingleJet40")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet40_Barrel", myvec(1,"L1_SingleJet40_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet50", myvec(1,"L1_SingleJet50")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet60", myvec(1,"L1_SingleJet60")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet6", myvec(1,"L1_SingleJet6")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet6_Barrel", myvec(1,"L1_SingleJet6_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet6_Central", myvec(1,"L1_SingleJet6_Central")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleJet6_Endcap", myvec(1,"L1_SingleJet6_Endcap")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu0", myvec(1,"L1_SingleMu0")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu10", myvec(1,"L1_SingleMu10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu14", myvec(1,"L1_SingleMu14")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu3", myvec(1,"L1_SingleMu3")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu5", myvec(1,"L1_SingleMu5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMu7", myvec(1,"L1_SingleMu7")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMuBeamHalo", myvec(1,"L1_SingleMuBeamHalo")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleMuOpen", myvec(1,"L1_SingleMuOpen")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet10", myvec(1,"L1_SingleTauJet10")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet10_Barrel", myvec(1,"L1_SingleTauJet10_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet20", myvec(1,"L1_SingleTauJet20")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet20_Barrel", myvec(1,"L1_SingleTauJet20_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet30", myvec(1,"L1_SingleTauJet30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet30_Barrel", myvec(1,"L1_SingleTauJet30_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet50", myvec(1,"L1_SingleTauJet50")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet8", myvec(1,"L1_SingleTauJet8")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_SingleTauJet8_Barrel", myvec(1,"L1_SingleTauJet8_Barrel")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TauJet10_ETM30", myvec(1,"L1_TauJet10_ETM30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TauJet10_ETM40", myvec(1,"L1_TauJet10_ETM40")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TripleJet30", myvec(1,"L1_TripleJet30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("L1_TripleMu3", myvec(1,"L1_TripleMu3")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenL1_ZeroBias", myvec(1,"OpenL1_ZeroBias")));

  /* New Taus */
  // This is openhlt and not standard hlt,
  // but the same mechanism can also be used here!
  // L1 prescales can be checked in the same way as 
  // for standard hlt in CheckOpenHlt(). 
  // Look for "New Taus" in OHltTreeOpen.cpp!
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet30");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20", vtmp));

  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  //vtmp.push_back("L1_SingleTauJet50"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk5", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk5", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  //vtmp.push_back("L1_SingleTauJet50"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk10", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk10", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet20"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet20"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_Trk5", vtmp));

  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_L2R", vtmp));

  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  //vtmp.push_back("L1_SingleTauJet50"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk5_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk5_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  //vtmp.push_back("L1_SingleTauJet50"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk10_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk10_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet20"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_L2R", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet20"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_Trk5_L2R", vtmp));

  // L3 iso
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk5_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk5_L2R_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk10_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet30"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau30_Trk10_L2R_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet20"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_Trk5_L3I", vtmp));

  vtmp.clear();
  vtmp.push_back("L1_DoubleTauJet20"); vtmp.push_back("L1_DoubleJet40");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleLooseIsoTau15_Trk5_L2R_L3I", vtmp));

  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk5_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk5_L2R_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk10_L3I", vtmp));
  
  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleJet60");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleLooseIsoTau20_Trk10_L2R_L3I", vtmp));
  
  vtmp.clear(); 
  vtmp.push_back("L1_SingleTauJet20"); vtmp.push_back("L1_SingleMu14"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Mu14", vtmp)); 

  vtmp.clear();
  vtmp.push_back("L1_SingleTauJet40"); vtmp.push_back("L1_SingleJet100");   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SingleIsoTau20_Trk5", vtmp));

  // Jets
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Jet15", myvec(1,"L1_SingleJet15"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Jet30", myvec(1,"L1_SingleJet15")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Jet50", myvec(1,"L1_SingleJet30"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Jet80", myvec(1,"L1_SingleJet50")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Jet110", myvec(1,"L1_SingleJet70")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Jet180", myvec(1,"L1_SingleJet70")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_QuadJet30", myvec(1,"L1_SingleJet15"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_FwdJet20", myvec(1,"L1_IsoEG10_Jet15_ForJet10"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DiJetAve15U", myvec(1,"L1_SingleJet15"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DiJetAve30U", myvec(1,"L1_SingleJet30")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DiJetAve50U", myvec(1,"L1_SingleJet50")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DiJetAve70U", myvec(1,"L1_SingleJet70")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DiJetAve130U", myvec(1,"L1_SingleJet70")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_SumET120", myvec(1,"L1_ETT60")));   

  // Egammas
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1SingleEG5",  myvec(1,"L1_SingleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele10_SW_L1R",  myvec(1,"L1_SingleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele15_SW_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele15_SW_LooseTrackIso_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele20_SW_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleEle10_SW_L1R",  myvec(1,"L1_DoubleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele15_SC15_SW_LooseTrackIso_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele20_SC15_SW_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele25_SW_L1R",  myvec(1,"L1_SingleEG8")));

  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon10_L1R",  myvec(1,"L1_SingleEG5")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon10_LooseEcalIso_TrackIso_L1R",  myvec(1,"L1_SingleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon15_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon20_LooseEcalIso_TrackIso_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon25_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon25_LooseEcalIso_TrackIso_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon30_L1R",  myvec(1,"L1_SingleEG8"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoublePhoton10_L1R",  myvec(1,"L1_DoubleEG5"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoublePhoton15_L1R",  myvec(1,"L1_DoubleEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoublePhoton15_VeryLooseEcalIso_L1R",  myvec(1,"L1_DoubleEG5")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Photon70_L1R",  myvec(1,"L1_SingleEG8")));  

  // Muons
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_IsoMu3", myvec(1,"L1_SingleMu3"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_IsoMu9", myvec(1,"L1_SingleMu7")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Mu5", myvec(1,"L1_SingleMu3")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Mu9", myvec(1,"L1_SingleMu7")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Mu11", myvec(1,"L1_SingleMu10")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Mu15", myvec(1,"L1_SingleMu10")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L2Mu11", myvec(1,"L1_SingleMu7")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Mu20", myvec(1,"L1_SingleMu20")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L2Mu9_1Jet30", myvec(1,"L1_Mu5_Jet6")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleMu0", myvec(1,"L1_DoubleMuOpen")));  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_DoubleMu3", myvec(1,"L1_DoubleMu3")));   
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1DoubleMuOpen", myvec(1,"L1_DoubleMuOpen")));   

  vtmp.clear();  
  vtmp.push_back("L1_SingleMuOpen"); vtmp.push_back("L1_SingleMu3"); vtmp.push_back("L1_SingleMu5");  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1MuOpen", vtmp));  

  vtmp.clear();   
  vtmp.push_back("L1_DoubleMu3"); vtmp.push_back("L1_SingleMu7"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Mu", vtmp));   

  // MET
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1MET20", myvec(1,"L1_ETM20")));    
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_MET25", myvec(1,"L1_ETM20")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_MET50", myvec(1,"L1_ETM40")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_MET100", myvec(1,"L1_ETM80")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_HT300_MHT100", myvec(1,"L1_HTT200")));     
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_HT250", myvec(1,"L1_HTT200")));     

  // b-tagging
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_BTagIP_Jet80", myvec(1,"L1_SingleJet70")));
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_BTagIP_Jet120", myvec(1,"L1_SingleJet70"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_BTagMu_Jet20", myvec(1,"L1_Mu3_Jet15"))); 

  // Cross-channel
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_Ele10_SW_L1R_TripleJet30", myvec(1,"L1_EG5_TripleJet15"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L2Mu9_DiJet30", myvec(1,"L1_Mu5_SingleJet15"))); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L2Mu5_Photon9_L1R", myvec(1,"L1_Mu3_EG5"))); 

  vtmp.clear();    
  vtmp.push_back("L1_SingleEG10"); vtmp.push_back("L1_SingleMu14");  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Mu14_L1SingleEG10", vtmp)); 
  
  vtmp.clear();    
  vtmp.push_back("L1_SingleJet15"); vtmp.push_back("L1_SingleMu14");  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Mu14_L1SingleJet15", vtmp)); 
  
  vtmp.clear();    
  vtmp.push_back("L1_ETM30"); vtmp.push_back("L1_SingleMu14");  
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_L1Mu14_L1ETM30", vtmp)); 
  
  // Minbias
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_ZeroBias", myvec(1,"OpenL1_ZeroBias")));

  vtmp.clear();      
  vtmp.push_back("L1_SingleHfBitCountsRing1_1"); vtmp.push_back("L1_DoubleHfBitCountsRing1_P1N1"); 
  vtmp.push_back("L1_SingleHfRingEtSumsRing1_4"); vtmp.push_back("L1_DoubleHfRingEtSumsRing1_P4N4"); 
  vtmp.push_back("L1_SingleHfRingEtSumsRing2_4"); vtmp.push_back("L1_DoubleHfRingEtSumsRing2_P4N4"); 
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_MinBiasHcal", vtmp)); 

  vtmp.clear();      
  vtmp.push_back("L1_SingleEG1"); vtmp.push_back("L1_DoubleEG1"); vtmp.push_back("L1_SingleEG2");
  map_L1SeedsOfStandardHLTPath.insert(mypair("OpenHLT_MinBiasEcal", vtmp)); 

}
