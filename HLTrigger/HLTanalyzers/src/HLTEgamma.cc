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
  
  const int kMaxPixEl = 10000;
  pixelpt = new float[kMaxPixEl];
  pixelphi = new float[kMaxPixEl];
  pixeleta = new float[kMaxPixEl];
  pixelet = new float[kMaxPixEl];
  pixele = new float[kMaxPixEl];
  const int kMaxSilEl = 10000;
  silelpt = new float[kMaxSilEl];
  silelphi = new float[kMaxSilEl];
  sileleta = new float[kMaxSilEl];
  silelet = new float[kMaxSilEl];
  silele = new float[kMaxSilEl];
  const int kMaxPhot = 10000;
  photonpt = new float[kMaxPhot];
  photonphi = new float[kMaxPhot];
  photoneta = new float[kMaxPhot];
  photonet = new float[kMaxPhot];
  photone = new float[kMaxPhot];

  // Egamma-specific branches of the tree 
  HltTree->Branch("NobjPixElectron",&npixele,"NobjPixElectron/I");
  HltTree->Branch("ElectronPxPt",pixelpt,"ElectronPxPt[NobjPixElectron]/F");
  HltTree->Branch("ElectronPxPhi",pixelphi,"ElectronPxPhi[NobjPixElectron]/F");
  HltTree->Branch("ElectronPxEta",pixeleta,"ElectronPxEta[NobjPixElectron]/F");
  HltTree->Branch("ElectronPxEt",pixelet,"ElectronPxEt[NobjPixElectron]/F");
  HltTree->Branch("ElectronPxE",pixele,"ElectronPxE[NobjPixElectron]/F");
  HltTree->Branch("NobjSilElectron",&nsilele,"NobjSilElectron/I");
  HltTree->Branch("ElectronSiPt",silelpt,"ElectronSiPt[NobjSilElectron]/F");
  HltTree->Branch("ElectronSiPhi",silelphi,"ElectronSiPhi[NobjSilElectron]/F");
  HltTree->Branch("ElectronSiEta",sileleta,"ElectronSiEta[NobjSilElectron]/F");
  HltTree->Branch("ElectronSiEt",silelet,"ElectronSiEt[NobjSilElectron]/F");
  HltTree->Branch("ElectronSiE",silele,"ElectronSiE[NobjSilElectron]/F");
  HltTree->Branch("NobjPhoton",&nphoton,"NobjPhoton/I");
  HltTree->Branch("PhotonPt",photonpt,"PhotonPt[NobjPhoton]/F");
  HltTree->Branch("PhotonPhi",photonphi,"PhotonPhi[NobjPhoton]/F");
  HltTree->Branch("PhotonEta",photoneta,"PhotonEta[NobjPhoton]/F");
  HltTree->Branch("PhotonEt",photonet,"PhtonEt[NobjPhoton]/F");
  HltTree->Branch("PhotonE",photone,"PhotonE[NobjPhoton]/F");

}

/* **Analyze the event** */
void HLTEgamma::analyze(const ElectronCollection& pixElectron,
			const ElectronCollection& silElectron,
			const PhotonCollection& Photon,
			const CaloGeometry& geom,
			TTree* HltTree) {

  //std::cout << " Beginning HLTEgamma " << std::endl;

  if (&pixElectron) {
    ElectronCollection mypixelectrons;
    mypixelectrons=pixElectron;
    npixele = mypixelectrons.size();
    std::sort(mypixelectrons.begin(),mypixelectrons.end(),EtGreater());
    typedef ElectronCollection::const_iterator ceiter;
    int ipixel=0;
    for (ceiter i=mypixelectrons.begin(); i!=mypixelectrons.end(); i++) {
      pixelpt[ipixel] = i->pt();
      pixelphi[ipixel] = i->phi();
      pixeleta[ipixel] = i->eta();
      pixelet[ipixel] = i->et();
      pixele[ipixel] = i->energy();
      ipixel++;
    }
  }
  else {npixele = 0;}

  if (&silElectron) {
    ElectronCollection mysilelectrons;
    mysilelectrons=silElectron;
    nsilele = mysilelectrons.size();
    std::sort(mysilelectrons.begin(),mysilelectrons.end(),EtGreater());
    typedef ElectronCollection::const_iterator seiter;
    int isil=0;
    for (seiter i=mysilelectrons.begin(); i!=mysilelectrons.end(); i++) {
      silelpt[isil] = i->pt();
      silelphi[isil] = i->phi();
      sileleta[isil] = i->eta();
      silelet[isil] = i->et();
      silele[isil] = i->energy();
      isil++;
    }
  }
  else {nsilele = 0;}

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
