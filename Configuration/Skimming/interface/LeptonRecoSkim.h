// Class:      LeptonRecoSkim
// 
/**\class LeptonRecoSkim LeptonRecoSkim.cc Configuration/Skimming/src/LeptonRecoSkim.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Massimiliano Chiorboli,40 4-A01,+41227671535,
//         Created:  Wed Mar 31 21:49:08 CEST 2010
// $Id: LeptonRecoSkim.h,v 1.1 2010/11/05 18:37:50 torimoto Exp $
//
//
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

//includes for reco objects
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//includes for PF
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


//includes for Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"


class LeptonRecoSkim : public edm::EDFilter {
 public:
  explicit LeptonRecoSkim(const edm::ParameterSet&);
  ~LeptonRecoSkim();
  
 private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void handleObjects(const edm::Event&, const edm::EventSetup& iSetup);



  edm::InputTag hltLabel;
  std::string filterName;
  edm::InputTag m_electronSrc;
  edm::InputTag m_pfelectronSrc;
  edm::InputTag m_muonSrc;
  edm::InputTag m_jetsSrc;
  edm::InputTag m_pfjetsSrc;
  edm::InputTag m_ebRecHitsSrc;
  edm::InputTag m_eeRecHitsSrc;

  const reco::GsfElectronCollection*           theElectronCollection  ;    
  const reco::PFCandidateCollection*           thePfCandidateCollection  ;    
  const reco::MuonCollection*                  theMuonCollection      ;
  const reco::CaloJetCollection*               theCaloJetCollection   ;
  const reco::PFJetCollection*                 thePFJetCollection   ;
  const EcalRecHitCollection*            theEcalBarrelCollection;
  const EcalRecHitCollection*            theEcalEndcapCollection;
  const CaloTopology*                    theCaloTopology;
  const CaloGeometry*                    theCaloGeometry;

  bool firstEvent;


  
  bool useElectronSelection;
  bool usePfElectronSelection;
  bool useMuonSelection;
  bool useHtSelection;
  bool usePFHtSelection;


  bool ElectronCutPassed;
  bool PfElectronCutPassed;
  bool MuonCutPassed;
  bool HtCutPassed;
  bool PFHtCutPassed;
  

  double ptElecMin;
  double ptPfElecMin;
  int nSelectedElectrons;
  int nSelectedPfElectrons;
  double ptGlobalMuonMin;
  double ptTrackerMuonMin;
  int nSelectedMuons;
  double htMin;
  double pfHtMin;
  double htJetThreshold;
  double pfHtJetThreshold;



  int NeventsTotal;
  int NeventsFiltered;
  int NHltMu9;
  int NHltDiMu3;

  int NtotalElectrons;
  int NmvaElectrons;
  // ----------member data ---------------------------
};

