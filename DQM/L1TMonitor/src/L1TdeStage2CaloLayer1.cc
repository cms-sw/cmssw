/*
 * \file L1TdeStage2CaloLayer1.cc
 *
 * N. Smith <nick.smith@cern.ch>
 */
//Modified by Bhawna Gomber <bhawna.gomber@cern.ch>

#include "DQM/L1TMonitor/interface/L1TdeStage2CaloLayer1.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"


L1TdeStage2CaloLayer1::L1TdeStage2CaloLayer1(const edm::ParameterSet & ps) :
  dataLabel_(ps.getParameter<edm::InputTag>("dataSource")),
  dataSource_(consumes<CaloTowerBxCollection>(dataLabel_)),
  emulLabel_(ps.getParameter<edm::InputTag>("emulSource")),
  emulSource_(consumes<CaloTowerBxCollection>(emulLabel_)),
  hcalTowers_(consumes<HcalTrigPrimDigiCollection>(edm::InputTag("l1tCaloLayer1Digis"))),
  histFolder_(ps.getParameter<std::string>("histFolder")),
  tpFillThreshold_(ps.getUntrackedParameter<int>("etDistributionsFillThreshold", 0))
{
}

L1TdeStage2CaloLayer1::~L1TdeStage2CaloLayer1()
{
}

void L1TdeStage2CaloLayer1::dqmBeginRun(const edm::Run&, const edm::EventSetup&)
{
}

void L1TdeStage2CaloLayer1::analyze(const edm::Event & event, const edm::EventSetup & es)
{
  edm::Handle<CaloTowerBxCollection> dataTowers;
  event.getByToken(dataSource_, dataTowers);
  edm::Handle<CaloTowerBxCollection> emulTowers;
  event.getByToken(emulSource_, emulTowers);
  edm::Handle<HcalTrigPrimDigiCollection> hcalTowers;
  event.getByToken(hcalTowers_, hcalTowers);

  // We'll fill sets, compare, and then dissect comparison failures after.
  SimpleTowerSet dataTowerSet;
  // BXVector::begin(int bx)
  for ( auto iter = dataTowers->begin(0); iter != dataTowers->end(0); ++iter) {
    const auto& tower = *iter;
    int eta = tower.hwEta();
    if ( eta == 29 ) eta = 30;
    if ( eta == -29 ) eta = -30;
    dataTowerSet.emplace(eta, tower.hwPhi(), tower.hwPt() + (tower.hwEtRatio()<<9) + (tower.hwQual()<<12), true);
    if ( tower.hwPt() > tpFillThreshold_ ) {
      dataOcc_->Fill(eta, tower.hwPhi());
      dataEtDistribution_->Fill(tower.hwPt());
    }
  }
  SimpleTowerSet emulTowerSet;
  for ( auto iter = emulTowers->begin(0); iter != emulTowers->end(0); ++iter) {
    const auto& tower = *iter;
    emulTowerSet.emplace(tower.hwEta(), tower.hwPhi(), tower.hwPt() + (tower.hwEtRatio()<<9) + (tower.hwQual()<<12), false);
    if ( tower.hwPt() > tpFillThreshold_ ) {
      emulOcc_->Fill(tower.hwEta(), tower.hwPhi());
      emulEtDistribution_->Fill(tower.hwPt());
    }
  }
  
  if ( dataTowerSet.size() != emulTowerSet.size() ) {
    edm::LogError("L1TdeStage2CaloLayer1") << "Data and Emulation have different number of trigger towers! data=" << dataTowerSet.size() << ", emul=" << emulTowerSet.size() << std::endl;
    towerCountMismatchesPerBx_->Fill(event.bunchCrossing());
    return;
  }

  auto dataIter = dataTowerSet.begin();
  auto emulIter = emulTowerSet.begin();
  while ( dataIter != dataTowerSet.end() && emulIter != emulTowerSet.end() ) {
    auto dataTower = *(dataIter++);
    auto emulTower = *(emulIter++);
    assert(dataTower.ieta_==emulTower.ieta_ && dataTower.iphi_==emulTower.iphi_);

    etCorrelation_->Fill(dataTower.et(), emulTower.et());

    if ( abs(dataTower.ieta_) >= 30 ) {
      fbCorrelationHF_->Fill(dataTower.fb(), emulTower.fb());
    }

    if ( dataTower.data_ == emulTower.data_ ) {
      // Perfect match
      if ( dataTower.et() > tpFillThreshold_ ) {
        matchOcc_->Fill(dataTower.ieta_, dataTower.iphi_);
      }
    } else {
      // Ok, now dissect the failure
      if ( dataTower.et() != emulTower.et() ) {
        if      ( dataTower.et() == 0 ) failureOccEtDataZero_->Fill(dataTower.ieta_, dataTower.iphi_);
        else if ( emulTower.et() == 0 ) failureOccEtEmulZero_->Fill(dataTower.ieta_, dataTower.iphi_);
        else failureOccEtMismatch_->Fill(dataTower.ieta_, dataTower.iphi_);

        etMismatchDiff_->Fill(dataTower.et() - emulTower.et());
        etMismatchByLumi_->Fill(event.id().luminosityBlock());
        updateMismatch(event, 0);
      }
      if ( dataTower.er() != emulTower.er() ) {
        failureOccErMismatch_->Fill(dataTower.ieta_, dataTower.iphi_);
        erMismatchByLumi_->Fill(event.id().luminosityBlock());
        updateMismatch(event, 1);
      }
      if ( dataTower.fb() != emulTower.fb() ) {
        failureOccFbMismatch_->Fill(dataTower.ieta_, dataTower.iphi_);
        fbMismatchByLumi_->Fill(event.id().luminosityBlock());
        fbMismatchesPerBx_->Fill(event.bunchCrossing());
        dataEtDistributionFBMismatch_->Fill(dataTower.et());
        updateMismatch(event, 2);
      }
    }
  }
}

void L1TdeStage2CaloLayer1::updateMismatch(const edm::Event& e, int mismatchType) {
  auto id = e.id();
  std::string eventString{std::to_string(id.run()) + ":" + std::to_string(id.luminosityBlock()) + ":" + std::to_string(id.event())};
  if ( last20MismatchArray_.at(lastMismatchIndex_).first == eventString ) {
    // same event
    last20MismatchArray_.at(lastMismatchIndex_).second |= 1<<mismatchType;
  }
  else {
    // New event, advance
    lastMismatchIndex_ = (lastMismatchIndex_+1) % 20;
    last20MismatchArray_.at(lastMismatchIndex_) = {eventString, 1<<mismatchType};
  }
}

void L1TdeStage2CaloLayer1::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {
  // Ugly way to loop backwards through the last 20 mismatches
  for (size_t ibin=1, imatch=lastMismatchIndex_; ibin<=20; ibin++, imatch=(imatch+19)%20) {
    last20Mismatches_->getTH2F()->GetYaxis()->SetBinLabel(ibin, last20MismatchArray_.at(imatch).first.c_str());
    for(int itype=0; itype<4; ++itype) {
      int binContent = (last20MismatchArray_.at(imatch).second>>itype) & 1;
      last20Mismatches_->setBinContent(itype+1, ibin, binContent);
    }
  }
}

void L1TdeStage2CaloLayer1::endLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup&) {
  auto id = static_cast<double>(lumi.id().luminosityBlock()); // uint64_t
  // Simple way to embed current lumi to auto-scale axis limits in render plugin
  etMismatchByLumi_->getTH1F()->SetBinContent(0, id);
  erMismatchByLumi_->getTH1F()->SetBinContent(0, id);
  fbMismatchByLumi_->getTH1F()->SetBinContent(0, id);
}

void L1TdeStage2CaloLayer1::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run , const edm::EventSetup& es) 
{
  auto bookEt = [&ibooker](std::string name, std::string title) {
    return ibooker.book1D(name, title, 256, -0.5, 255.5);
  };
  auto bookEtDiff = [&ibooker](std::string name, std::string title) {
    return ibooker.book1D(name, title, 511, -255.5, 255.5);
  };
  auto bookEtCorrelation = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title, 256, -0.5, 255.5, 256, -0.5, 255.5);
  };
  auto bookOccupancy = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title, 83, -41.5, 41.5, 72, 0.5, 72.5);
  };


  ibooker.setCurrentFolder(histFolder_+"/Occupancies");

  dataOcc_ = bookOccupancy("dataOcc", "Tower Occupancy for Data");
  emulOcc_ = bookOccupancy("emulOcc", "Tower Occupancy for Emulator");
  matchOcc_ = bookOccupancy("matchOcc", "Tower Occupancy for Data/Emulator Full Matches");
  failureOccEtMismatch_ = bookOccupancy("failureOccEtMismatch", "Tower Occupancy for Data/Emulator ET Mismatch");
  failureOccEtDataZero_ = bookOccupancy("failureOccEtDataZero", "Tower Occupancy for Data ET Zero, Emul Nonzero");
  failureOccEtEmulZero_ = bookOccupancy("failureOccEtEmulZero", "Tower Occupancy for Data ET Nonzero, Emul Zero");
  failureOccErMismatch_ = bookOccupancy("failureOccErMismatch", "Tower Occupancy for Data/Emulator ET Ratio Mismatch");
  failureOccFbMismatch_ = bookOccupancy("failureOccFbMismatch", "Tower Occupancy for Data/Emulator Feature Bit Mismatch");

  ibooker.setCurrentFolder(histFolder_+"/EtDistributions");
  dataEtDistribution_ = bookEt("dataEtDistribution", "ET distribution for towers in data");
  dataEtDistributionFBMismatch_ = bookEt("dataEtDistributionFBMismatch", "ET distribution for towers in data when FB Mismatch");
  emulEtDistribution_ = bookEt("emulEtDistribution", "ET distribution for towers in emulator");
  etCorrelation_ = bookEtCorrelation("EtCorrelation", "Et correlation for all towers;Data tower Et;Emulator tower Et");
  matchEtDistribution_ = bookEt("matchEtDistribution", "ET distribution for towers matched between data and emulator");
  etMismatchDiff_ = bookEtDiff("etMismatchDiff", "ET difference (data-emulator) for ET mismatches");
  fbCorrelationHF_ = ibooker.book2D("FbCorrelationHF", "Feature Bit correlation for HF;Data;Emulator", 16, -0.5, 15.5, 16, -0.5, 15.5);

  ibooker.setCurrentFolder(histFolder_+"/Mismatch");

  const int nLumis = 2000;
  etMismatchByLumi_  = ibooker.book1D("etMismatchByLumi", "ET Mismatch counts per lumi section;LumiSection;Counts", nLumis, .5, nLumis+0.5);
  erMismatchByLumi_  = ibooker.book1D("erMismatchByLumi", "ET Ratio Mismatch counts per lumi section;LumiSection;Counts", nLumis, .5, nLumis+0.5);
  fbMismatchByLumi_  = ibooker.book1D("fbMismatchByLumi", "Feature Bit Mismatch counts per lumi section;LumiSection;Counts", nLumis, .5, nLumis+0.5);

  fbMismatchesPerBx_   = ibooker.book1D("fbMismatchesPerBx", "Feature Bit Mismatch counts per bunch crossing", 3564, -.5, 3563.5);
  towerCountMismatchesPerBx_ = ibooker.book1D("towerCountMismatchesPerBx", "CaloTower size mismatch counts per bunch crossing", 3564, -.5, 3563.5);

  last20Mismatches_ = ibooker.book2D("last20Mismatches", 
                                             "Log of last 20 mismatches (use json tool to copy/paste)",
                                             4, 0, 4, 20, 0, 20);
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(1, "Et Mismatch");
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(2, "Et ratio Mismatch");
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(3, "Feature bit Mismatch");
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(4, "-");
  for (size_t i=0; i<20; ++i) last20MismatchArray_.at(i) = {"-", 0};
  for (size_t i=1; i<=20; ++i) last20Mismatches_->getTH2F()->GetYaxis()->SetBinLabel(i, "-");
}

