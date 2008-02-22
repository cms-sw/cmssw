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

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "HLTrigger/HLTanalyzers/interface/HLTEgamma.h"


HLTEgamma::HLTEgamma() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTEgamma::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myEmParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }
  
  const int kMaxEl = 10000;
  elpt = new float[kMaxEl];
  elphi = new float[kMaxEl];
  eleta = new float[kMaxEl];
  elet = new float[kMaxEl];
  ele = new float[kMaxEl];
  const int kMaxPhot = 10000;
  photonpt = new float[kMaxPhot];
  photonphi = new float[kMaxPhot];
  photoneta = new float[kMaxPhot];
  photonet = new float[kMaxPhot];
  photone = new float[kMaxPhot];

  // Egamma-specific branches of the tree 
  HltTree->Branch("NrecoElec",&nele,"NrecoElec/I");
  HltTree->Branch("recoElecPt",elpt,"recoElecPt[NrecoElec]/F");
  HltTree->Branch("recoElecPhi",elphi,"recoElecPhi[NrecoElec]/F");
  HltTree->Branch("recoElecEta",eleta,"recoElecEta[NrecoElec]/F");
  HltTree->Branch("recoElecEt",elet,"recoElecEt[NrecoElec]/F");
  HltTree->Branch("recoElecE",ele,"recoElecE[NrecoElec]/F");
  HltTree->Branch("NrecoPhot",&nphoton,"NrecoPhot/I");
  HltTree->Branch("recoPhotPt",photonpt,"recoPhotPt[NrecoPhot]/F");
  HltTree->Branch("recoPhotPhi",photonphi,"recoPhotPhi[NrecoPhot]/F");
  HltTree->Branch("recoPhotEta",photoneta,"recoPhotEta[NrecoPhot]/F");
  HltTree->Branch("recoPhotEt",photonet,"recoPhotEt[NrecoPhot]/F");
  HltTree->Branch("recoPhotE",photone,"recoPhotE[NrecoPhot]/F");

}

/* **Analyze the event** */
void HLTEgamma::analyze(const ElectronCollection& Electron,
			const PhotonCollection& Photon,
			const CaloGeometry& geom,
			TTree* HltTree) {

  //std::cout << " Beginning HLTEgamma " << std::endl;

  if (&Electron) {
    ElectronCollection myelectrons;
    myelectrons=Electron;
    nele = myelectrons.size();
    std::sort(myelectrons.begin(),myelectrons.end(),EtGreater());
    typedef ElectronCollection::const_iterator ceiter;
    int iel=0;
    for (ceiter i=myelectrons.begin(); i!=myelectrons.end(); i++) {
      elpt[iel] = i->pt();
      elphi[iel] = i->phi();
      eleta[iel] = i->eta();
      elet[iel] = i->et();
      ele[iel] = i->energy();
      iel++;
    }
  }
  else {nele = 0;}

  if (&Photon) {
    PhotonCollection myphotons;
    myphotons=Photon;
    nphoton = myphotons.size();
    std::sort(myphotons.begin(),myphotons.end(),EtGreater());
    typedef PhotonCollection::const_iterator phiter;
    int ipho=0;
    for (phiter i=myphotons.begin(); i!=myphotons.end(); i++) {
      photonpt[ipho] = i->pt();
      photonphi[ipho] = i->phi();
      photoneta[ipho] = i->eta();
      photonet[ipho] = i->et();
      photone[ipho] = i->energy();
      ipho++;
    }
  }
  else {nphoton = 0;}

}
