#ifndef DQMHcalIsoTrackAlCaReco_H
#define DQMHcalIsoTrackAlCaReco_H

// -*- C++ -*-
//
// Package:    DQMOffline/CalibCalo
// Class:      DQMHcalIsoTrackAlCaReco
// 
/**\class DQMHcalIsoTrackAlCaReco DQMHcalIsoTrackAlCaReco.cc DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV
//         Created:  Tue Oct  14 16:10:31 CEST 2008
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <fstream>

#include "TH1F.h"

class DQMHcalIsoTrackAlCaReco : public edm::EDAnalyzer {
public:
  explicit DQMHcalIsoTrackAlCaReco(const edm::ParameterSet&);
  ~DQMHcalIsoTrackAlCaReco();
  
  
private:

  DQMStore* dbe_;  

  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  std::string folderName_;
  bool saveToFile_;
  std::string outRootFileName_;
  edm::EDGetTokenT<trigger::TriggerEvent> hltEventTag_;
  std::string l1FilterTag_;
  std::vector<std::string> hltFilterTag_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> arITrLabel_;
  edm::InputTag recoTrLabel_;
  double pThr_;
  double heLow_;
  double heUp_;
  
  MonitorElement* hl3Pt;
  MonitorElement* hl3eta;
  MonitorElement* hl3AbsEta;
  MonitorElement* hl3phi;
  MonitorElement* hOffL3TrackMatch;
  MonitorElement* hOffL3TrackPtRat;

  MonitorElement* hOffP_0005;
  MonitorElement* hOffP_0510;
  MonitorElement* hOffP_1015;
  MonitorElement* hOffP_1520;

  MonitorElement* hOffP;

  MonitorElement* hTracksSumP;
  MonitorElement* hTracksMaxP;

  MonitorElement* hDeposEcalInnerEB;
  MonitorElement* hDeposEcalOuterEB;
  MonitorElement* hDeposEcalInnerEE;
  MonitorElement* hDeposEcalOuterEE;
  
  MonitorElement* hL1jetMatch;

  MonitorElement* hOffEtaFP;
  MonitorElement* hOffAbsEta;
  MonitorElement* hOffPhiFP;

  MonitorElement* hOffEta;
  MonitorElement* hOffPhi;
  
  MonitorElement* hOccupancyFull;
  MonitorElement* hOccupancyHighEn;

  MonitorElement* hPurityEta;
  MonitorElement* hPurityPhi;

  int nTotal;
  int nHLTL3accepts;
  int nameLength_;
  int l1nameLength_;
  
  std::pair<int, int> towerIndex(double eta, double phi);

};

#endif
