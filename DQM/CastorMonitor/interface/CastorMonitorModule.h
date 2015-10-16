#ifndef CastorMonitorModule_H
#define CastorMonitorModule_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimG4CMS/Calo/interface/CaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Provenance/interface/EventID.h"  

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"
#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"
#include "RecoJets/JetProducers/plugins/CastorJetIDProducer.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h" //-- no CastorUnpackerReport at the moment !
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h" //-- HcalCastorDetId

//#include "DQM/CastorMonitor/interface/CastorMonitorSelector.h"
#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorChannelQualityMonitor.h"
#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorPSMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorHIMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorDataIntegrityMonitor.h"
//#include "DQM/CastorMonitor/interface/CastorTowerJetMonitor.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h" //-- use to get/dump Calib to DB 
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h" //-- use to get/hold channel status
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"


//// #include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h" //-- 

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>



class CastorMonitorModule : public DQMEDAnalyzer{

public:
  
  CastorMonitorModule(const edm::ParameterSet& ps);
  ~CastorMonitorModule();
  
protected:
  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& eventSetup);
  
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& eventSetup) ;

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& eventSetup);
  
  void endRun(const edm::Run& run, const edm::EventSetup& eventSetup);

private:

  int fVerbosity;  
  float fedsUnpacked;
  bool rawOK_    ;
  bool reportOK_ ;
  bool digiOK_   ;
  bool rechitOK_ ;
  
  int irun_,ilumisec_,ievent_,itime_,ibunch_;
  std::string rootFolder_;

  int ievt_;
  int NBunchesOrbit;
  edm::EDGetTokenT<FEDRawDataCollection> inputTokenRaw_;
  edm::EDGetTokenT<HcalUnpackerReport> inputTokenReport_;
  edm::EDGetTokenT<CastorDigiCollection> inputTokenDigi_;
  edm::EDGetTokenT<CastorRecHitCollection> inputTokenRecHitCASTOR_;
   typedef std::vector<reco::CastorTower> CastorTowerCollection;
  edm::EDGetTokenT<CastorTowerCollection> inputTokenCastorTowers_;
   typedef std::vector<reco::BasicJet> BasicJetCollection;
  edm::EDGetTokenT<BasicJetCollection> JetAlgorithm;

//  edm::InputTag inputLabelCastorTowers_;
//  edm::InputTag JetAlgorithm;
//  edm::InputTag trigResultsSource;

  CastorRecHitMonitor*      RecHitMon_;
  CastorDigiMonitor*        DigiMon_;
  CastorLEDMonitor*         LedMon_;

  MonitorElement* CastorEventProduct;

  edm::ESHandle<CastorDbService> conditions_;

  bool showTiming_; 
  edm::CPUTimer cpu_timer; 
  edm::ESHandle<CastorPedestals> dbPedestals;

};

#endif
