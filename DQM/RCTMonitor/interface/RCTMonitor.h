
#ifndef RCTMonitor_RCTMonitor_H
#define RCTMOnitor_RCTMonitor_H

// -*- C++ -*-
//
// Package:    RCTMonitor
// Class:      RCTMonitor
//
/**\class RCTMonitor

 Description: DQM monitor for the Regional Calorimeter Trigger

*/
//
// Original Author:  S.Dasu. H.Patel, A.Savin
// version 0 is based on the GCTMonitor package created by A.Tapper
//
//

// Framework files

#include "FWCore/PluginManager/interface/ModuleDef.h"

#include <iostream>
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

#include <TH1F.h>
#include <TH1I.h>

//#include <SimDataFormats/Track/interface/SimTrackContainer.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM files
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
//#include <SimDataFormats/Track/interface/SimTrackContainer.h>

// TPs
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

struct rct_location {
  unsigned crate, card, region;
};

class RCTMonitor : public DQMEDAnalyzer {
 public:
  explicit RCTMonitor(const edm::ParameterSet&);
  ~RCTMonitor();
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void FillRCT(const edm::Event&, const edm::EventSetup&);

 private:
  // RCT stuff
  MonitorElement* m_rctRegionsEtEtaPhi;
  MonitorElement* m_rctRegionsOccEtaPhi;
  MonitorElement* m_rctTauVetoEtaPhi;
  MonitorElement* m_rctRegionEt;

  MonitorElement* m_rctIsoEmRankEtaPhi1;
  MonitorElement* m_rctIsoEmRankEtaPhi10;
  MonitorElement* m_rctIsoEmOccEtaPhi1;
  MonitorElement* m_rctIsoEmOccEtaPhi10;
  MonitorElement* m_rctNonIsoEmRankEtaPhi1;
  MonitorElement* m_rctNonIsoEmRankEtaPhi10;
  MonitorElement* m_rctRelaxedEmRankEtaPhi1;
  MonitorElement* m_rctRelaxedEmRankEtaPhi10;
  MonitorElement* m_rctNonIsoEmOccEtaPhi1;
  MonitorElement* m_rctNonIsoEmOccEtaPhi10;
  MonitorElement* m_rctRelaxedEmOccEtaPhi1;
  MonitorElement* m_rctRelaxedEmOccEtaPhi10;
  MonitorElement* m_rctIsoEmRank1;
  MonitorElement* m_rctIsoEmRank10;
  MonitorElement* m_rctRelaxedEmRank1;
  MonitorElement* m_rctRelaxedEmRank10;
  MonitorElement* m_rctNonIsoEmRank1;
  MonitorElement* m_rctNonIsoEmRank10;

  // Bins etc.
  // GCT and RCT
  static const unsigned int ETABINS;
  static const float ETAMIN;
  static const float ETAMAX;
  static const unsigned int PHIBINS;
  static const float PHIMIN;
  static const float PHIMAX;
  static const unsigned int METPHIBINS;
  static const float METPHIMIN;
  static const float METPHIMAX;
  static const unsigned int R6BINS;
  static const float R6MIN;
  static const float R6MAX;
  static const unsigned int R10BINS;
  static const float R10MIN;
  static const float R10MAX;
  static const unsigned int R12BINS;
  static const float R12MIN;
  static const float R12MAX;

  // HCAL and ECAL TPs
  static const unsigned int TPETABINS;
  static const float TPETAMIN;
  static const float TPETAMAX;
  static const unsigned int TPPHIBINS;
  static const float TPPHIMIN;
  static const float TPPHIMAX;
  static const unsigned int RTPBINS;
  static const float RTPMIN;
  static const float RTPMAX;

  // Physical bins 1 GeV to 1 TeV in steps of 1 GeV
  static const unsigned int TEVBINS;
  static const float TEVMIN;
  static const float TEVMAX;
  static const unsigned int L1EETABINS;
  static const float L1EETAMIN;
  static const float L1EETAMAX;
  static const unsigned int L1EPHIBINS;
  static const float L1EPHIMIN;
  static const float L1EPHIMAX;

  // define Token(-s)
  edm::EDGetTokenT<L1CaloEmCollection> m_rctSourceToken_;
};

#endif
