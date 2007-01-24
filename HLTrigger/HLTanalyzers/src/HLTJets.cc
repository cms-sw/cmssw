#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/HLTJets.h"

HLTJets::HLTJets() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTJets::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myJetParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myJetParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myJetParams.getParameter<bool>( *iParam );
  }

  const int kMaxJetCal = 10000;
  jcalpt = new float[kMaxJetCal];
  jcalphi = new float[kMaxJetCal];
  jcaleta = new float[kMaxJetCal];
  jcalet = new float[kMaxJetCal];
  jcale = new float[kMaxJetCal];
  const int kMaxJetgen = 10000;
  jgenpt = new float[kMaxJetgen];
  jgenphi = new float[kMaxJetgen];
  jgeneta = new float[kMaxJetgen];
  jgenet = new float[kMaxJetgen];
  jgene = new float[kMaxJetgen];
  const int kMaxTower = 10000;
  towet = new float[kMaxTower];
  toweta = new float[kMaxTower];
  towphi = new float[kMaxTower];
  towen = new float[kMaxTower];
  towem = new float[kMaxTower];
  towhd = new float[kMaxTower];
  towoe = new float[kMaxTower];

  // Jet- MEt-specific branches of the tree 
  HltTree->Branch("NobjJetCal",&njetcal,"NobjJetCal/I");
  HltTree->Branch("NobjJetGen",&njetgen,"NobjJetGen/I");
  HltTree->Branch("NobjTowCal",&ntowcal,"NobjTowCal/I");
  HltTree->Branch("JetCalPt",jcalpt,"JetCalPt[NobjJetCal]/F");
  HltTree->Branch("JetCalPhi",jcalphi,"JetCalPhi[NobjJetCal]/F");
  HltTree->Branch("JetCalEta",jcaleta,"JetCalEta[NobjJetCal]/F");
  HltTree->Branch("JetCalEt",jcalet,"JetCalEt[NobjJetCal]/F");
  HltTree->Branch("JetCalE",jcale,"JetCalE[NobjJetCal]/F");
  HltTree->Branch("JetGenPt",jgenpt,"JetGenPt[NobjJetGen]/F");
  HltTree->Branch("JetGenPhi",jgenphi,"JetGenPhi[NobjJetGen]/F");
  HltTree->Branch("JetGenEta",jgeneta,"JetGenEta[NobjJetGen]/F");
  HltTree->Branch("JetGenEt",jgenet,"JetGenEt[NobjJetGen]/F");
  HltTree->Branch("JetGenE",jgene,"JetGenE[NobjJetGen]/F");
  HltTree->Branch("TowEt",towet,"TowEt[NobjTowCal]/F");
  HltTree->Branch("TowEta",toweta,"TowEta[NobjTowCal]/F");
  HltTree->Branch("TowPhi",towphi,"TowPhi[NobjTowCal]/F");
  HltTree->Branch("TowE",towen,"TowE[NobjTowCal]/F");
  HltTree->Branch("TowEm",towem,"TowEm[NobjTowCal]/F");
  HltTree->Branch("TowHad",towhd,"TowHad[NobjTowCal]/F");
  HltTree->Branch("TowOE",towoe,"TowOE[NobjTowCal]/F");
  HltTree->Branch("MetCal",&mcalmet,"MetCal/F");
  HltTree->Branch("MetCalPhi",&mcalphi,"MetCalPhi/F");
  HltTree->Branch("MetCalSum",&mcalsum,"MetCalSum/F");
  HltTree->Branch("MetGen",&mgenmet,"MetGen/F");
  HltTree->Branch("MetGenPhi",&mgenphi,"MetGenPhi/F");
  HltTree->Branch("MetGenSum",&mgensum,"MetGenSum/F");

  //for(int ieta=0;ieta<NETA;ieta++){cout << " ieta " << ieta << " eta min " << CaloTowerEtaBoundries[ieta] <<endl;}

}

/* **Analyze the event** */
void HLTJets::analyze(const CaloJetCollection& calojets,
		      const GenJetCollection& genjets,
		      const CaloMETCollection& recmets,
		      const GenMETCollection& genmets,
		      const CaloTowerCollection& caloTowers,
		      const CaloGeometry& geom,
		      TTree* HltTree) {

  //std::cout << " Beginning HLTJets " << std::endl;

  if (&calojets) {
    CaloJetCollection mycalojets;
    mycalojets=calojets;
    std::sort(mycalojets.begin(),mycalojets.end(),PtGreater());
    njetcal = mycalojets.size();
    typedef CaloJetCollection::const_iterator cjiter;
    int jcal=0;
    for ( cjiter i=mycalojets.begin(); i!=mycalojets.end(); i++) {
      jcalpt[jcal] = i->pt();
      jcalphi[jcal] = i->phi();
      jcaleta[jcal] = i->eta();
      jcalet[jcal] = i->et();
      jcale[jcal] = i->energy();
      jcal++;
    }
  }
  else {njetcal = 0;}

  if (&caloTowers){
    ntowcal = caloTowers.size();
    int jtow = 0;
    for ( CaloTowerCollection::const_iterator tower=caloTowers.begin(); tower!=caloTowers.end(); tower++) {
      towet[jtow] = tower->et();
      toweta[jtow] = tower->eta();
      towphi[jtow] = tower->phi();
      towen[jtow] = tower->energy();
      towem[jtow] = tower->emEnergy();
      towhd[jtow] = tower->hadEnergy();
      towoe[jtow] = tower->outerEnergy();
      jtow++;
    }
  }
  else {ntowcal = 0;}

  if (&recmets) {
    typedef CaloMETCollection::const_iterator cmiter;
    for ( cmiter i=recmets.begin(); i!=recmets.end(); i++) {
      mcalmet = i->pt();
      mcalphi = i->phi();
      mcalsum = i->sumEt();
    }
  }

  if (_Monte){

    if (&genjets) {
      GenJetCollection mygenjets;
      mygenjets=genjets;
      std::sort(mygenjets.begin(),mygenjets.end(),PtGreater());
      njetgen = mygenjets.size();
      typedef GenJetCollection::const_iterator gjiter;
      int jgen=0;
      for ( gjiter i=mygenjets.begin(); i!=mygenjets.end(); i++) {
	jgenpt[jgen] = i->pt();
	jgenphi[jgen] = i->phi();
	jgeneta[jgen] = i->eta();
	jgenet[jgen] = i->et();
	jgene[jgen] = i->energy();
	jgen++;
      }
    }
    else {njetgen = 0;}
    
    if (&genmets) {
      typedef GenMETCollection::const_iterator gmiter;
      for ( gmiter i=genmets.begin(); i!=genmets.end(); i++) {
	mgenmet = i->pt();
	mgenphi = i->phi();
	mgensum = i->sumEt();
      }
    }

  }


}
