#ifndef CutSelector_h
#define CutSelector_h

/*  \class CutSelector
*
*  Class to apply analysis cuts in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: August 2007
*
*/
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

class CutSelector {

 public:
  
  //Constructor
  CutSelector(edm::ParameterSet userCut_params);
  //Destructor
  virtual ~CutSelector(){};

  //Methods
  void handleObjects(const edm::Event&);
  bool isSelected(const edm::Event&);

 private:
  
  // Define the parameters
  std::string m_electronSrc;
  std::string m_muonSrc;
  std::string m_jetsSrc;
  std::string m_photonSrc;
  std::string m_calometSrc;
  double user_metMin;
  double user_ptJet1Min;
  double user_ptJet2Min;
  double user_ptElecMin;
  double user_ptMuonMin;
  double user_ptPhotMin;

  const reco::PixelMatchGsfElectronCollection* theElectronCollection;  
  const reco::MuonCollection*                  theMuonCollection    ;
  const reco::PhotonCollection*                thePhotonCollection  ;
  const reco::CaloJetCollection*               theCaloJetCollection ;
  const reco::CaloMETCollection*               theCaloMETCollection ;

};

#endif
