#ifndef CastorMonitorModule_H
#define CastorMonitorModule_H

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHit.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"  //no CastorUnpackerReport at the moment

//#include "DQM/CastorMonitor/interface/CastorMonitorSelector.h"
#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorChannelQualityMonitor.h"
#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorTowerJetMonitor.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"    //-- use to get/dump Calib to DB
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"  //-- use to get/hold channel status
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sys/time.h>
#include <vector>

class CastorMonitorModule : public DQMOneEDAnalyzer<> {
public:
  CastorMonitorModule(const edm::ParameterSet &ps);
  ~CastorMonitorModule() override;

protected:
  void analyze(const edm::Event &iEvent, const edm::EventSetup &) override;

  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, const edm::EventSetup &) override;

  void dqmEndRun(const edm::Run &run, const edm::EventSetup &) override;

private:
  int fVerbosity;
  std::string subsystemname_;
  //  int NBunchesOrbit;
  int ievt_;
  //  edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1tStage2uGtSource_;//for L1
  //  uGT DAQ readout record edm::EDGetTokenT<GlobalAlgBlkBxCollection>
  //  TokenL1TStage2uGtSource;

  edm::EDGetTokenT<edm::TriggerResults> tokenTriggerResults;
  edm::EDGetTokenT<FEDRawDataCollection> inputTokenRaw_;
  edm::EDGetTokenT<HcalUnpackerReport> inputTokenReport_;
  edm::EDGetTokenT<CastorDigiCollection> inputTokenDigi_;
  edm::EDGetTokenT<CastorRecHitCollection> inputTokenRecHitCASTOR_;
  typedef std::vector<reco::CastorTower> CastorTowerCollection;
  edm::EDGetTokenT<CastorTowerCollection> inputTokenCastorTowers_;
  typedef std::vector<reco::BasicJet> BasicJetCollection;
  edm::EDGetTokenT<BasicJetCollection> JetAlgorithm;

  edm::ESGetToken<CastorDbService, CastorDbRecord> castorDbServiceToken_;

  std::unique_ptr<CastorRecHitMonitor> RecHitMon_;
  std::unique_ptr<CastorDigiMonitor> DigiMon_;
  std::unique_ptr<CastorLEDMonitor> LedMon_;

  //  MonitorElement* algoBits_before_bxmask_bx_inEvt;
  //  MonitorElement* algoBits_before_bxmask_bx_global;
  MonitorElement *CastorEventProduct;
  MonitorElement *hunpkrep;

  bool showTiming_;
  edm::CPUTimer cpu_timer;
};

#endif
