#ifndef RecoSelector_h
#define RecoSelector_h

/*  \class RecoSelector
*
*  Class to apply analysis cuts in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: August 2007
//         Maurizio Pierini
//         Maria Spiropulu
*
*/
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

class RecoSelector {

 public:
  
  //Constructor
  RecoSelector(edm::ParameterSet userCut_params);
  //Destructor
  virtual ~RecoSelector(){};

  //Methods
  void handleObjects(const edm::Event&);
  bool isSelected(const edm::Event&);
  std::string GetName();

 private:
  
  // Define the parameters
  std::string name;
  std::string m_electronSrc;
  std::string m_muonSrc;
  std::string m_jetsSrc;
  std::string m_photonSrc;
  std::string m_photonProducerSrc;
  std::string m_calometSrc;
  double reco_metMin;
  double reco_ptJet1Min;
  double reco_ptJet2Min;
  double reco_ptElecMin;
  double reco_ptMuonMin;
  double reco_ptPhotMin;

  const reco::GsfElectronCollection* theElectronCollection;  
  const reco::MuonCollection*                  theMuonCollection    ;
  const reco::PhotonCollection*                thePhotonCollection  ;
  const reco::CaloJetCollection*               theCaloJetCollection ;
  const reco::CaloMETCollection*               theCaloMETCollection ;

};

#endif
