#include <DQM/CastorMonitor/interface/CastorMonitorModule.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

int NLumiSec = 0;
int TrigNameIndex = 0;

CastorMonitorModule::CastorMonitorModule(const edm::ParameterSet& ps) {
  fVerbosity = ps.getUntrackedParameter<int>("debug", 0);
  subsystemname_ =
      ps.getUntrackedParameter<std::string>("subSystemFolder", "Castor");
  inputTokenRaw_ = consumes<FEDRawDataCollection>(
      ps.getParameter<edm::InputTag>("rawLabel"));
  inputTokenReport_ = consumes<HcalUnpackerReport>(
      ps.getParameter<edm::InputTag>("unpackerReportLabel"));
  inputTokenDigi_ = consumes<CastorDigiCollection>(
      ps.getParameter<edm::InputTag>("digiLabel"));
  inputTokenRecHitCASTOR_ = consumes<CastorRecHitCollection>(
      ps.getParameter<edm::InputTag>("CastorRecHitLabel"));
  inputTokenCastorTowers_ = consumes<CastorTowerCollection>(
      ps.getParameter<edm::InputTag>("CastorTowerLabel"));
  JetAlgorithm = consumes<BasicJetCollection>(
      ps.getParameter<edm::InputTag>("CastorBasicJetsLabel"));
  tokenTriggerResults = consumes<edm::TriggerResults>(
      ps.getParameter<edm::InputTag>("tagTriggerResults"));

  showTiming_ = ps.getUntrackedParameter<bool>("showTiming", false);

  irun_ = ilumisec_ = ievent_ = ibunch_ = 0;
  TrigNameIndex = 0;

  DigiMon_ = nullptr;
  RecHitMon_ = nullptr;
  LedMon_ = nullptr;

  if (ps.getUntrackedParameter<bool>("DigiMonitor", false))
    DigiMon_ = new CastorDigiMonitor(ps);

  if (ps.getUntrackedParameter<bool>("RecHitMonitor", false))
    RecHitMon_ = new CastorRecHitMonitor(ps);

  if (ps.getUntrackedParameter<bool>("LEDMonitor", false))
    LedMon_ = new CastorLEDMonitor(ps);

  ievt_ = 0;
}

CastorMonitorModule::~CastorMonitorModule() {
  if (DigiMon_ != nullptr) delete DigiMon_;
  if (RecHitMon_ != nullptr) delete RecHitMon_;
  if (LedMon_ != nullptr) delete LedMon_;
}

void CastorMonitorModule::dqmBeginRun(const edm::Run& iRun,
                                      const edm::EventSetup& iSetup) {
  if (fVerbosity > 0) LogPrint("CastorMonitorModule") << "dqmBeginRun(start)";
  NLumiSec = 0;

  iSetup.get<CastorDbRecord>().get(conditions_);
}

void CastorMonitorModule::bookHistograms(DQMStore::IBooker& ibooker,
                                         const edm::Run& iRun,
                                         const edm::EventSetup& iSetup) {
  if (DigiMon_ != nullptr) {
    DigiMon_->bookHistograms(ibooker, iRun, iSetup);
  }
  if (RecHitMon_ != nullptr) {
    RecHitMon_->bookHistograms(ibooker, iRun, iSetup);
  }
  if (LedMon_ != nullptr) {
    LedMon_->bookHistograms(ibooker, iRun, iSetup);
  }

  ibooker.setCurrentFolder(subsystemname_);
  char s[60];
  sprintf(s, "CastorEventProducts");
  CastorEventProduct = ibooker.book1DD(s, s, 6, -0.5, 5.5);
  CastorEventProduct->getTH1D()->GetYaxis()->SetTitle("Events");
  TAxis* xa = CastorEventProduct->getTH1D()->GetXaxis();
  xa->SetBinLabel(1, "FEDs/3");
  xa->SetBinLabel(2, "RawData");
  xa->SetBinLabel(3, "Digi");
  xa->SetBinLabel(4, "RecHits");
  xa->SetBinLabel(5, "Towers");
  xa->SetBinLabel(6, "Jets");

  sprintf(s, "CASTORUnpackReport");
  hunpkrep = ibooker.bookProfile(s, s, 6, -0.5, 5.5, 100, 0, 1.e10, "");
  xa = hunpkrep->getTProfile()->GetXaxis();
  xa->SetBinLabel(1, "N_FEDs");
  xa->SetBinLabel(2, "SPIGOT_Err");
  xa->SetBinLabel(3, "empty");
  xa->SetBinLabel(4, "busy");
  xa->SetBinLabel(5, "OvF");
  xa->SetBinLabel(6, "BadDigis");
  return;
}

void CastorMonitorModule::beginLuminosityBlock(
    const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
  ++NLumiSec;
  if (fVerbosity > 0)
    LogPrint("CastorMonitorModule")
        << "beginLuminosityBlock(start): " << NLumiSec << "(" << ilumisec_
        << ")";
}

void CastorMonitorModule::endLuminosityBlock(
    const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {}

void CastorMonitorModule::endRun(const edm::Run& r,
                                 const edm::EventSetup& context) {
  if (DigiMon_ != nullptr) {
    DigiMon_->endRun();
  }
}

void CastorMonitorModule::analyze(const edm::Event& iEvent,
                                  const edm::EventSetup& eventSetup) {
  if (fVerbosity > 1) LogPrint("CastorMonitorModule") << "analyze (start)";

  irun_ = iEvent.id().run();
  ilumisec_ = iEvent.luminosityBlock();
  ievent_ = iEvent.id().event();
  ibunch_ = iEvent.bunchCrossing();

  ievt_++;

  bool rawOK_ = true;
  bool digiOK_ = true;
  bool rechitOK_ = true, towerOK_ = true, jetsOK_ = true;
  int nDigi = 0, nrecHits = 0, nTowers = 0, nJets = 0;

  edm::Handle<edm::TriggerResults> TrigResults;
  iEvent.getByToken(tokenTriggerResults, TrigResults);

  edm::Handle<FEDRawDataCollection> RawData;
  iEvent.getByToken(inputTokenRaw_, RawData);
  if (!RawData.isValid()) rawOK_ = false;

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
    LogPrint("CastorMonitorModule")
        << "CastorProductValid(size): RawDataValid=" << RawData.isValid()
        << " Digi=" << digiOK_ << "(" << nDigi << ") Hits=" << rechitOK_ << "("
        << nrecHits << ")"
        << " Towers=" << towerOK_ << "(" << nTowers << ")"
        << " Jets=" << jetsOK_ << "(" << nJets << ")";

  CastorEventProduct->Fill(0, fedsUnpacked / 3.);
  CastorEventProduct->Fill(1, rawOK_);
  CastorEventProduct->Fill(2, digiOK_);
  CastorEventProduct->Fill(3, rechitOK_);
  CastorEventProduct->Fill(4, towerOK_);
  CastorEventProduct->Fill(5, jetsOK_);

  if (digiOK_)
    DigiMon_->processEvent(iEvent, *CastorDigi, *TrigResults, *conditions_);
  if (showTiming_) {
    cpu_timer.stop();
    if (DigiMon_ != nullptr)
      std::cout << "TIMER:: DIGI MONITOR ->" << cpu_timer.cpuTime()
                << std::endl;
    cpu_timer.reset();
    cpu_timer.start();
  }

  if (rechitOK_) RecHitMon_->processEvent(*CastorHits);
  if (showTiming_) {
    cpu_timer.stop();
    if (RecHitMon_ != nullptr)
      std::cout << "TIMER:: RECHIT MONITOR->" << cpu_timer.cpuTime()
                << std::endl;
    cpu_timer.reset();
    cpu_timer.start();
  }

  if (towerOK_) RecHitMon_->processEventTowers(*castorTowers);
  if (jetsOK_) RecHitMon_->processEventJets(*jets);

  if (fVerbosity > 0 && ievt_ % 100 == 0)
    LogPrint("CastorMonitorModule") << "processed " << ievt_ << " events";
  return;
}

DEFINE_FWK_MODULE(CastorMonitorModule);
