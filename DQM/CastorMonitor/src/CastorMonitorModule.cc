#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/CastorMonitor/interface/CastorMonitorModule.h>
//#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include <string>

//**************************************************************//
//***************** CastorMonitorModule       ******************//
//***************** Author: Dmytro Volyanskyy ******************//
//***************** Date  : 22.11.2008 (first version) *********//
////---- simple event filter which directs events to monitoring tasks:
////---- access unpacked data from each event and pass them to monitoring tasks
////---- revision: 06.10.2010 (Dima Volyanskyy)
////---- last revision: 31.05.2011 (Panos Katsas)
////---- LS1 upgrade: 04.06.2013 (Pedro Cipriano)
//**************************************************************//

//---- critical revision 26.06.2014 (Vladimir Popov)

//**************************************************************//

using namespace std;
using namespace edm;

CastorMonitorModule::CastorMonitorModule(const edm::ParameterSet &ps)
    : castorDbServiceToken_{esConsumes<CastorDbService, CastorDbRecord>()} {
  fVerbosity = ps.getUntrackedParameter<int>("debug", 0);
  subsystemname_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "Castor");
  inputTokenRaw_ = consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("rawLabel"));
  inputTokenReport_ = consumes<HcalUnpackerReport>(ps.getParameter<edm::InputTag>("unpackerReportLabel"));
  inputTokenDigi_ = consumes<CastorDigiCollection>(ps.getParameter<edm::InputTag>("digiLabel"));
  inputTokenRecHitCASTOR_ = consumes<CastorRecHitCollection>(ps.getParameter<edm::InputTag>("CastorRecHitLabel"));
  inputTokenCastorTowers_ = consumes<CastorTowerCollection>(ps.getParameter<edm::InputTag>("CastorTowerLabel"));
  JetAlgorithm = consumes<BasicJetCollection>(ps.getParameter<edm::InputTag>("CastorBasicJetsLabel"));
  tokenTriggerResults = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("tagTriggerResults"));

  showTiming_ = ps.getUntrackedParameter<bool>("showTiming", false);

  if (ps.getUntrackedParameter<bool>("DigiMonitor", false))
    DigiMon_ = std::make_unique<CastorDigiMonitor>(ps, consumesCollector());

  if (ps.getUntrackedParameter<bool>("RecHitMonitor", false))
    RecHitMon_ = std::make_unique<CastorRecHitMonitor>(ps);

  if (ps.getUntrackedParameter<bool>("LEDMonitor", false))
    LedMon_ = std::make_unique<CastorLEDMonitor>(ps);

  ievt_ = 0;
}

CastorMonitorModule::~CastorMonitorModule() {}

void CastorMonitorModule::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &) {
  if (fVerbosity > 0)
    LogPrint("CastorMonitorModule") << "dqmBeginRun(start)";
}

void CastorMonitorModule::bookHistograms(DQMStore::IBooker &ibooker,
                                         const edm::Run &iRun,
                                         const edm::EventSetup &iSetup) {
  if (DigiMon_) {
    // Run histos only since there is endRun processing.
    auto scope = DQMStore::IBooker::UseRunScope(ibooker);
    DigiMon_->bookHistograms(ibooker, iRun, iSetup);
  }
  if (RecHitMon_) {
    RecHitMon_->bookHistograms(ibooker, iRun);
  }
  if (LedMon_) {
    LedMon_->bookHistograms(ibooker, iRun);
  }

  ibooker.setCurrentFolder(subsystemname_);
  char s[60];
  sprintf(s, "CastorEventProducts");
  CastorEventProduct = ibooker.book1DD(s, s, 6, -0.5, 5.5);
  CastorEventProduct->setAxisTitle("Events", /* axis */ 2);
  CastorEventProduct->setBinLabel(1, "FEDs/3");
  CastorEventProduct->setBinLabel(2, "RawData");
  CastorEventProduct->setBinLabel(3, "Digi");
  CastorEventProduct->setBinLabel(4, "RecHits");
  CastorEventProduct->setBinLabel(5, "Towers");
  CastorEventProduct->setBinLabel(6, "Jets");

  sprintf(s, "CASTORUnpackReport");
  hunpkrep = ibooker.bookProfile(s, s, 6, -0.5, 5.5, 100, 0, 1.e10, "");
  hunpkrep->setBinLabel(1, "N_FEDs");
  hunpkrep->setBinLabel(2, "SPIGOT_Err");
  hunpkrep->setBinLabel(3, "empty");
  hunpkrep->setBinLabel(4, "busy");
  hunpkrep->setBinLabel(5, "OvF");
  hunpkrep->setBinLabel(6, "BadDigis");
  return;
}

void CastorMonitorModule::dqmEndRun(const edm::Run &r, const edm::EventSetup &) {
  if (DigiMon_) {
    DigiMon_->endRun();
  }
}

void CastorMonitorModule::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (fVerbosity > 1)
    LogPrint("CastorMonitorModule") << "analyze (start)";

  ievt_++;

  bool rawOK_ = true;
  bool digiOK_ = true;
  bool rechitOK_ = true, towerOK_ = true, jetsOK_ = true;
  int nDigi = 0, nrecHits = 0, nTowers = 0, nJets = 0;

  edm::Handle<edm::TriggerResults> TrigResults;
  iEvent.getByToken(tokenTriggerResults, TrigResults);

  edm::Handle<FEDRawDataCollection> RawData;
  iEvent.getByToken(inputTokenRaw_, RawData);
  if (!RawData.isValid())
    rawOK_ = false;

  float fedsUnpacked = 0.;
  edm::Handle<HcalUnpackerReport> report;
  iEvent.getByToken(inputTokenReport_, report);
  if (!report.isValid())
    rawOK_ = false;
  else {
    const std::vector<int> feds = (*report).getFedsUnpacked();
    fedsUnpacked = float(feds.size());
    hunpkrep->Fill(0, fedsUnpacked);
    hunpkrep->Fill(1, report->spigotFormatErrors());
    hunpkrep->Fill(2, report->emptyEventSpigots());
    hunpkrep->Fill(3, report->busySpigots());
    hunpkrep->Fill(4, report->OFWSpigots());
    hunpkrep->Fill(5, report->badQualityDigis());
  }

  edm::Handle<CastorDigiCollection> CastorDigi;
  iEvent.getByToken(inputTokenDigi_, CastorDigi);
  if (CastorDigi.isValid())
    nDigi = CastorDigi->size();
  else
    digiOK_ = false;

  edm::Handle<CastorRecHitCollection> CastorHits;
  iEvent.getByToken(inputTokenRecHitCASTOR_, CastorHits);
  if (CastorHits.isValid())
    nrecHits = CastorHits->size();
  else
    rechitOK_ = false;

  edm::Handle<reco::CastorTowerCollection> castorTowers;
  iEvent.getByToken(inputTokenCastorTowers_, castorTowers);
  if (castorTowers.isValid())
    nTowers = castorTowers->size();
  else
    towerOK_ = false;

  edm::Handle<reco::BasicJetCollection> jets;
  iEvent.getByToken(JetAlgorithm, jets);
  if (jets.isValid())
    nJets = jets->size();
  else
    jetsOK_ = false;

  if (fVerbosity > 0)
    LogPrint("CastorMonitorModule") << "CastorProductValid(size): RawDataValid=" << RawData.isValid()
                                    << " Digi=" << digiOK_ << "(" << nDigi << ") Hits=" << rechitOK_ << "(" << nrecHits
                                    << ")"
                                    << " Towers=" << towerOK_ << "(" << nTowers << ")"
                                    << " Jets=" << jetsOK_ << "(" << nJets << ")";

  CastorEventProduct->Fill(0, fedsUnpacked / 3.);
  CastorEventProduct->Fill(1, rawOK_);
  CastorEventProduct->Fill(2, digiOK_);
  CastorEventProduct->Fill(3, rechitOK_);
  CastorEventProduct->Fill(4, towerOK_);
  CastorEventProduct->Fill(5, jetsOK_);

  if (digiOK_) {
    const CastorDbService &conditions = iSetup.getData(castorDbServiceToken_);
    DigiMon_->processEvent(iEvent, *CastorDigi, *TrigResults, conditions);
  }
  if (showTiming_) {
    cpu_timer.stop();
    if (DigiMon_ != nullptr)
      std::cout << "TIMER:: DIGI MONITOR ->" << cpu_timer.cpuTime() << std::endl;
    cpu_timer.reset();
    cpu_timer.start();
  }

  if (rechitOK_)
    RecHitMon_->processEvent(*CastorHits);
  if (showTiming_) {
    cpu_timer.stop();
    if (RecHitMon_ != nullptr)
      std::cout << "TIMER:: RECHIT MONITOR->" << cpu_timer.cpuTime() << std::endl;
    cpu_timer.reset();
    cpu_timer.start();
  }

  if (towerOK_)
    RecHitMon_->processEventTowers(*castorTowers);
  if (jetsOK_)
    RecHitMon_->processEventJets(*jets);

  if (fVerbosity > 0 && ievt_ % 100 == 0)
    LogPrint("CastorMonitorModule") << "processed " << ievt_ << " events";
  return;
}

DEFINE_FWK_MODULE(CastorMonitorModule);
