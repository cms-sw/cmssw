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
//         Modified: Tue Mar   3 16:10:31 CEST 2015
//
//


// system include files
#include <memory>
#include <fstream>
#include <vector>

// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HcalIsolatedTrack/interface/HcalIsolatedTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/HcalIsolatedTrackCandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


class DQMHcalIsoTrackAlCaReco : public DQMEDAnalyzer {

public:
  explicit DQMHcalIsoTrackAlCaReco(const edm::ParameterSet&);
  ~DQMHcalIsoTrackAlCaReco();
  
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::string                             folderName_;
  std::vector<std::string>                l1FilterTag_, hltFilterTag_;
  std::vector<int>                        type_;
  edm::InputTag                           labelTrigger_, labelTrack_;
  edm::EDGetTokenT<trigger::TriggerEvent> tokTrigger_;
  edm::EDGetTokenT<reco::HcalIsolatedTrackCandidateCollection> tokTrack_;

  double                                  pThr_;
  
  std::vector<MonitorElement*>            hL1Pt_, hL1Eta_,  hL1phi_;
  std::vector<MonitorElement*>            hHltP_, hHltEta_, hHltPhi_;
  MonitorElement                         *hL3Dr_, *hL3Rat_;
  std::vector<MonitorElement*>            hOffP_;
  MonitorElement                         *hMaxP_, *hEnEcal_, *hIeta_, *hIphi_;

  int                                     nTotal_, nHLTaccepts_;
  std::vector<double>                     etaRange_;
  std::vector<unsigned int>               indexH_;
  std::vector<bool>                       ifL3_;
};

#endif
