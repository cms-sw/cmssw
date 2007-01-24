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

#include "HLTrigger/HLTanalyzers/interface/HLTMuon.h"

HLTMuon::HLTMuon() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTMuon::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myEmParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }

  const int kMaxMuon = 10000;
  muonpt = new float[kMaxMuon];
  muonphi = new float[kMaxMuon];
  muoneta = new float[kMaxMuon];
  muonet = new float[kMaxMuon];
  muone = new float[kMaxMuon];

  // Muon-specific branches of the tree 
  HltTree->Branch("NobjMuon",&nmuon,"NobjMuon/I");
  HltTree->Branch("MuonPt",muonpt,"MuonPt[NobjMuon]/F");
  HltTree->Branch("MuonPhi",muonphi,"MuonPhi[NobjMuon]/F");
  HltTree->Branch("MuonEta",muoneta,"MuonEta[NobjMuon]/F");
  HltTree->Branch("MuonEt",muonet,"MuonEt[NobjMuon]/F");
  HltTree->Branch("MuonE",muone,"MuonE[NobjMuon]/F");

}

/* **Analyze the event** */
void HLTMuon::analyze(const MuonCollection& Muon,
		      const CaloGeometry& geom,
		      TTree* HltTree) {

  //std::cout << " Beginning HLTMuon " << std::endl;

  if (&Muon) {
    MuonCollection mymuons;
    mymuons=Muon;
    std::sort(mymuons.begin(),mymuons.end(),PtGreater());
    nmuon = mymuons.size();
    typedef MuonCollection::const_iterator muiter;
    int imu=0;
    for (muiter i=mymuons.begin(); i!=mymuons.end(); i++) {
      muonpt[imu] = i->pt();
      muonphi[imu] = i->phi();
      muoneta[imu] = i->eta();
      muonet[imu] = i->et();
      muone[imu] = i->energy();
      imu++;
    }
  }
  else {nmuon = 0;}

}
