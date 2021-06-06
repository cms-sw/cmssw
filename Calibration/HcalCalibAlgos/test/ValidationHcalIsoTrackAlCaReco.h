// -*- C++ -*-
//
// Package:    Calibration/HcalCalibAlgos/plugins
// Class:      ValidationHcalIsoTrackAlCaReco
//
/**\class ValidationHcalIsoTrackAlCaReco ValidationHcalIsoTrackAlCaReco.cc Calibration/HcalCalibAlgos/plugins/ValidationHcalIsoTrackAlCaReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV, Sergey PETRUSHANKO
//         Created:  Tue Oct  14 16:10:31 CEST 2008
//
//

// system include files
#include <memory>

// user include files

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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

// Sergey +

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

// Sergey -

#include <fstream>

#include "TH1F.h"

class ValidationHcalIsoTrackAlCaReco : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;
  explicit ValidationHcalIsoTrackAlCaReco(const edm::ParameterSet&);
  ~ValidationHcalIsoTrackAlCaReco();

private:
  DQMStore* dbe_;

  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  std::string folderName_;
  bool saveToFile_;
  std::string outRootFileName_;
  edm::InputTag hltFilterTag_;
  edm::InputTag recoTrLabel_;

  edm::EDGetTokenT<trigger::TriggerEvent> tok_hlt_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> tok_arITr_;
  edm::EDGetTokenT<edm::SimTrackContainer> tok_simTrack_;

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

  MonitorElement* hDeposEcalInner;
  MonitorElement* hDeposEcalOuter;

  MonitorElement* hOffEtaFP;
  MonitorElement* hOffAbsEta;
  MonitorElement* hOffPhiFP;

  MonitorElement* hOffEta;
  MonitorElement* hOffPhi;

  MonitorElement* hOccupancyFull;
  MonitorElement* hOccupancyHighEn;

  MonitorElement* hPurityEta;
  MonitorElement* hPurityPhi;

  // Sergey +

  MonitorElement* hSimPt;
  MonitorElement* hSimPhi;
  MonitorElement* hSimEta;
  MonitorElement* hSimAbsEta;
  MonitorElement* hSimDist;
  MonitorElement* hSimPtRatOff;
  MonitorElement* hSimP;
  MonitorElement* hSimN;
  MonitorElement* hSimNN;
  MonitorElement* hSimNE;
  MonitorElement* hSimNM;

  // Sergey -

  int nTotal;
  int nHLTL3accepts;

  double getDist(double, double, double, double);

  // Sergey +

  double getDistInCM(double eta1, double phi1, double eta2, double phi2);

  // Sergey -

  std::pair<int, int> towerIndex(double eta, double phi);
};
