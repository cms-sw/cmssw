/*
 * \file L1TStage2CaloLayer1.cc
 *
 * N. Smith <nick.smith@cern.ch>
 */
//Modified by Bhawna Gomber <bhawna.gomber@cern.ch>
//Modified by Andrew Loeliger <andrew.loeliger@cern.ch>
//Modified by Ho-Fung Tsoi <ho.fung.tsoi@cern.ch>

#include "DQM/L1TMonitor/interface/L1TStage2CaloLayer1.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "EventFilter/L1TXRawToDigi/interface/UCTDAQRawData.h"
#include "EventFilter/L1TXRawToDigi/interface/UCTAMCRawData.h"

using namespace l1t;

L1TStage2CaloLayer1::L1TStage2CaloLayer1(const edm::ParameterSet& ps)
    : ecalTPSourceRecd_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecd"))),
      ecalTPSourceRecdLabel_(ps.getParameter<edm::InputTag>("ecalTPSourceRecd").label()),
      ecalTPSourceRecdBx1_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx1"))),
      ecalTPSourceRecdBx1Label_(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx1").label()),
      ecalTPSourceRecdBx2_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx2"))),
      ecalTPSourceRecdBx2Label_(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx2").label()),
      ecalTPSourceRecdBx3_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx3"))),
      ecalTPSourceRecdBx3Label_(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx3").label()),
      ecalTPSourceRecdBx4_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx4"))),
      ecalTPSourceRecdBx4Label_(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx4").label()),
      ecalTPSourceRecdBx5_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx5"))),
      ecalTPSourceRecdBx5Label_(ps.getParameter<edm::InputTag>("ecalTPSourceRecdBx5").label()),
      hcalTPSourceRecd_(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPSourceRecd"))),
      hcalTPSourceRecdLabel_(ps.getParameter<edm::InputTag>("hcalTPSourceRecd").label()),
      ecalTPSourceSent_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceSent"))),
      ecalTPSourceSentLabel_(ps.getParameter<edm::InputTag>("ecalTPSourceSent").label()),
      hcalTPSourceSent_(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPSourceSent"))),
      hcalTPSourceSentLabel_(ps.getParameter<edm::InputTag>("hcalTPSourceSent").label()),
      CaloTowerCollectionData_(
          consumes<l1t::CaloTowerBxCollection>(ps.getParameter<edm::InputTag>("CaloTowerCollectionData"))),
      CaloTowerCollectionDataLabel_(ps.getParameter<edm::InputTag>("CaloTowerCollectionData").label()),
      fedRawData_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("fedRawDataLabel"))),
      histFolder_(ps.getParameter<std::string>("histFolder")),
      tpFillThreshold_(ps.getUntrackedParameter<int>("etDistributionsFillThreshold", 0)),
      tpFillThreshold5Bx_(ps.getUntrackedParameter<int>("etDistributionsFillThreshold5Bx", 4)),
      ignoreHFfbs_(ps.getUntrackedParameter<bool>("ignoreHFfbs", false)) {}

L1TStage2CaloLayer1::~L1TStage2CaloLayer1() {}

void L1TStage2CaloLayer1::dqmAnalyze(const edm::Event& event,
                                     const edm::EventSetup& es,
                                     const CaloL1Information::monitoringDataHolder& eventMonitors) const {
  //This must be moved here compared to previous version to avoid a const function modifying a class variable
  //This unfortunately means that the variable is local and reallocated per event
  std::vector<std::pair<EcalTriggerPrimitiveDigi, EcalTriggerPrimitiveDigi>> ecalTPSentRecd_;
  std::vector<std::pair<HcalTriggerPrimitiveDigi, HcalTriggerPrimitiveDigi>> hcalTPSentRecd_;
  ecalTPSentRecd_.reserve(28 * 2 * 72);
  hcalTPSentRecd_.reserve(41 * 2 * 72);

  // Monitorables stored in Layer 1 raw data but
  // not accessible from existing persistent data formats
  edm::Handle<FEDRawDataCollection> fedRawDataCollection;
  event.getByToken(fedRawData_, fedRawDataCollection);
  bool caloLayer1OutOfRun{true};
  bool FATevent{false};
  bool additionalFB{false};
  if (fedRawDataCollection.isValid()) {
    caloLayer1OutOfRun = false;
    for (int iFed = 1354; iFed < 1360; iFed += 2) {
      const FEDRawData& fedRawData = fedRawDataCollection->FEDData(iFed);
      if (fedRawData.size() == 0) {
        caloLayer1OutOfRun = true;
        continue;  // In case one of 3 layer 1 FEDs not in
      }
      const uint64_t* fedRawDataArray = (const uint64_t*)fedRawData.data();
      UCTDAQRawData daqData(fedRawDataArray);
      for (uint32_t i = 0; i < daqData.nAMCs(); i++) {
        UCTAMCRawData amcData(daqData.amcPayload(i));
        const uint32_t* amcPtr = amcData.dataPtr();
        FATevent = ((amcPtr[5] >> 16) & 0xf) == 5;
        additionalFB = (amcPtr[5] >> 15) & 0x1;
        int lPhi = amcData.layer1Phi();
        if (daqData.BXID() != amcData.BXID()) {
          eventMonitors.bxidErrors_->Fill(lPhi);
        }
        if (daqData.L1ID() != amcData.L1ID()) {
          eventMonitors.l1idErrors_->Fill(lPhi);
        }
        // AMC payload header has 16 bit orbit number, AMC13 header is full 32
        if ((daqData.orbitNumber() & 0xFFFF) != amcData.orbitNo()) {
          eventMonitors.orbitErrors_->Fill(lPhi);
        }
      }
    }
  }

  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsSent;
  event.getByToken(ecalTPSourceSent_, ecalTPsSent);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecd;
  event.getByToken(ecalTPSourceRecd_, ecalTPsRecd);

  ecalTPSentRecd_.clear();

  ComparisonHelper::zip(ecalTPsSent->begin(),
                        ecalTPsSent->end(),
                        ecalTPsRecd->begin(),
                        ecalTPsRecd->end(),
                        std::inserter(ecalTPSentRecd_, ecalTPSentRecd_.begin()),
                        EcalTrigPrimDigiCollection::key_compare());

  int nEcalLinkErrors{0};
  int nEcalMismatch{0};

  for (const auto& tpPair : ecalTPSentRecd_) {
    auto sentTp = tpPair.first;
    if (sentTp.compressedEt() < 0) {
      // ECal zero-suppresses digis, and a default-constructed
      // digi has et=-1 apparently, but we know it should be zero
      EcalTriggerPrimitiveSample sample(0);
      EcalTriggerPrimitiveDigi tpg(sentTp.id());
      tpg.setSize(1);
      tpg.setSample(0, sample);
      swap(sentTp, tpg);
    }
    const auto& recdTp = tpPair.second;
    const int ieta = sentTp.id().ieta();
    const int iphi = sentTp.id().iphi();
    const bool towerMasked = recdTp.sample(0).raw() & (1 << 13);
    const bool linkMasked = recdTp.sample(0).raw() & (1 << 14);
    const bool linkError = recdTp.sample(0).raw() & (1 << 15);

    // Link status bits from layer 1
    if (towerMasked) {
      eventMonitors.ecalOccTowerMasked_->Fill(ieta, iphi);
    }
    if (linkMasked) {
      eventMonitors.ecalOccLinkMasked_->Fill(ieta, iphi);
    }

    if (sentTp.compressedEt() > tpFillThreshold_) {
      eventMonitors.ecalTPRawEtSent_->Fill(sentTp.compressedEt());
      eventMonitors.ecalOccSent_->Fill(ieta, iphi);
    }
    if (sentTp.fineGrain() == 1) {
      eventMonitors.ecalOccSentFgVB_->Fill(ieta, iphi);
    }

    if (towerMasked || caloLayer1OutOfRun) {
      // Do not compare if we have a mask applied
      continue;
    }

    if (linkError) {
      eventMonitors.ecalLinkError_->Fill(ieta, iphi);
      eventMonitors.ecalLinkErrorByLumi_->Fill(event.id().luminosityBlock());
      nEcalLinkErrors++;
      // Don't compare anymore, we already know its bad
      continue;
    }

    eventMonitors.ecalTPRawEtCorrelation_->Fill(sentTp.compressedEt(), recdTp.compressedEt());

    if (recdTp.compressedEt() > tpFillThreshold_) {
      eventMonitors.ecalTPRawEtRecd_->Fill(recdTp.compressedEt());
      eventMonitors.ecalOccupancy_->Fill(ieta, iphi);
      eventMonitors.ecalOccRecdEtWgt_->Fill(ieta, iphi, recdTp.compressedEt());
    }
    if (recdTp.fineGrain() == 1) {
      eventMonitors.ecalOccRecdFgVB_->Fill(ieta, iphi);
    }

    // Now for comparisons

    const bool EetAgreement = sentTp.compressedEt() == recdTp.compressedEt();
    const bool EfbAgreement = sentTp.fineGrain() == recdTp.fineGrain();
    if (EetAgreement && EfbAgreement) {
      // Full match
      if (sentTp.compressedEt() > tpFillThreshold_) {
        eventMonitors.ecalOccSentAndRecd_->Fill(ieta, iphi);
        eventMonitors.ecalTPRawEtSentAndRecd_->Fill(sentTp.compressedEt());
      }
    } else {
      // There is some issue
      eventMonitors.ecalDiscrepancy_->Fill(ieta, iphi);
      eventMonitors.ecalMismatchByLumi_->Fill(event.id().luminosityBlock());
      eventMonitors.ECALmismatchesPerBx_->Fill(event.bunchCrossing());
      nEcalMismatch++;

      if (not EetAgreement) {
        eventMonitors.ecalOccEtDiscrepancy_->Fill(ieta, iphi);
        eventMonitors.ecalTPRawEtDiffNoMatch_->Fill(recdTp.compressedEt() - sentTp.compressedEt());
        updateMismatch(event, 0, streamCache(event.streamID())->streamMismatchList);

        if (sentTp.compressedEt() == 0)
          eventMonitors.ecalOccRecdNotSent_->Fill(ieta, iphi);
        else if (recdTp.compressedEt() == 0)
          eventMonitors.ecalOccSentNotRecd_->Fill(ieta, iphi);
        else
          eventMonitors.ecalOccNoMatch_->Fill(ieta, iphi);
      }
      if (not EfbAgreement) {
        // occ for fine grain mismatch
        eventMonitors.ecalOccFgDiscrepancy_->Fill(ieta, iphi);
        updateMismatch(event, 1, streamCache(event.streamID())->streamMismatchList);
      }
    }
  }

  if (nEcalLinkErrors > streamCache(event.streamID())->streamNumMaxEvtLinkErrorsECAL)
    streamCache(event.streamID())->streamNumMaxEvtLinkErrorsECAL = nEcalLinkErrors;

  if (nEcalMismatch > streamCache(event.streamID())->streamNumMaxEvtMismatchECAL)
    streamCache(event.streamID())->streamNumMaxEvtMismatchECAL = nEcalMismatch;

  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecdBx1;
  event.getByToken(ecalTPSourceRecdBx1_, ecalTPsRecdBx1);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecdBx2;
  event.getByToken(ecalTPSourceRecdBx2_, ecalTPsRecdBx2);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecdBx3;
  event.getByToken(ecalTPSourceRecdBx3_, ecalTPsRecdBx3);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecdBx4;
  event.getByToken(ecalTPSourceRecdBx4_, ecalTPsRecdBx4);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecdBx5;
  event.getByToken(ecalTPSourceRecdBx5_, ecalTPsRecdBx5);

  for (const auto& tp : (*ecalTPsRecdBx1)) {
    if (tp.compressedEt() > tpFillThreshold5Bx_) {
      const int ieta = tp.id().ieta();
      const int iphi = tp.id().iphi();
      const bool towerMasked = tp.sample(0).raw() & (1 << 13);
      const bool linkError = tp.sample(0).raw() & (1 << 15);
      if (towerMasked || caloLayer1OutOfRun || linkError) {
        continue;
      }
      eventMonitors.ecalOccRecdBx1_->Fill(ieta, iphi);
      eventMonitors.ecalOccRecd5Bx_->Fill(1);
      eventMonitors.ecalOccRecd5BxEtWgt_->Fill(1, tp.compressedEt());
    }
  }
  for (const auto& tp : (*ecalTPsRecdBx2)) {
    if (tp.compressedEt() > tpFillThreshold5Bx_) {
      const int ieta = tp.id().ieta();
      const int iphi = tp.id().iphi();
      const bool towerMasked = tp.sample(0).raw() & (1 << 13);
      const bool linkError = tp.sample(0).raw() & (1 << 15);
      if (towerMasked || caloLayer1OutOfRun || linkError) {
        continue;
      }
      eventMonitors.ecalOccRecdBx2_->Fill(ieta, iphi);
      eventMonitors.ecalOccRecd5Bx_->Fill(2);
      eventMonitors.ecalOccRecd5BxEtWgt_->Fill(2, tp.compressedEt());
    }
  }
  for (const auto& tp : (*ecalTPsRecdBx3)) {
    if (tp.compressedEt() > tpFillThreshold5Bx_) {
      const int ieta = tp.id().ieta();
      const int iphi = tp.id().iphi();
      const bool towerMasked = tp.sample(0).raw() & (1 << 13);
      const bool linkError = tp.sample(0).raw() & (1 << 15);
      if (towerMasked || caloLayer1OutOfRun || linkError) {
        continue;
      }
      eventMonitors.ecalOccRecdBx3_->Fill(ieta, iphi);
      eventMonitors.ecalOccRecd5Bx_->Fill(3);
      eventMonitors.ecalOccRecd5BxEtWgt_->Fill(3, tp.compressedEt());
    }
  }
  for (const auto& tp : (*ecalTPsRecdBx4)) {
    if (tp.compressedEt() > tpFillThreshold5Bx_) {
      const int ieta = tp.id().ieta();
      const int iphi = tp.id().iphi();
      const bool towerMasked = tp.sample(0).raw() & (1 << 13);
      const bool linkError = tp.sample(0).raw() & (1 << 15);
      if (towerMasked || caloLayer1OutOfRun || linkError) {
        continue;
      }
      eventMonitors.ecalOccRecdBx4_->Fill(ieta, iphi);
      eventMonitors.ecalOccRecd5Bx_->Fill(4);
      eventMonitors.ecalOccRecd5BxEtWgt_->Fill(4, tp.compressedEt());
    }
  }
  for (const auto& tp : (*ecalTPsRecdBx5)) {
    if (tp.compressedEt() > tpFillThreshold5Bx_) {
      const int ieta = tp.id().ieta();
      const int iphi = tp.id().iphi();
      const bool towerMasked = tp.sample(0).raw() & (1 << 13);
      const bool linkError = tp.sample(0).raw() & (1 << 15);
      if (towerMasked || caloLayer1OutOfRun || linkError) {
        continue;
      }
      eventMonitors.ecalOccRecdBx5_->Fill(ieta, iphi);
      eventMonitors.ecalOccRecd5Bx_->Fill(5);
      eventMonitors.ecalOccRecd5BxEtWgt_->Fill(5, tp.compressedEt());
    }
  }

  edm::Handle<HcalTrigPrimDigiCollection> hcalTPsSent;
  event.getByToken(hcalTPSourceSent_, hcalTPsSent);
  edm::Handle<HcalTrigPrimDigiCollection> hcalTPsRecd;
  event.getByToken(hcalTPSourceRecd_, hcalTPsRecd);
  edm::Handle<l1t::CaloTowerBxCollection> caloTowerDataCol;
  event.getByToken(CaloTowerCollectionData_, caloTowerDataCol);

  hcalTPSentRecd_.clear();

  ComparisonHelper::zip(hcalTPsSent->begin(),
                        hcalTPsSent->end(),
                        hcalTPsRecd->begin(),
                        hcalTPsRecd->end(),
                        std::inserter(hcalTPSentRecd_, hcalTPSentRecd_.begin()),
                        HcalTrigPrimDigiCollection::key_compare());

  int nHcalLinkErrors{0};
  int nHcalMismatch{0};

  for (const auto& tpPair : hcalTPSentRecd_) {
    const auto& sentTp = tpPair.first;
    const auto& recdTp = tpPair.second;
    const int ieta = sentTp.id().ieta();
    if (abs(ieta) > 28 && sentTp.id().version() != 1)
      continue;
    const int iphi = sentTp.id().iphi();
    const bool towerMasked = recdTp.sample(0).raw() & (1 << 13);
    const bool linkMasked = recdTp.sample(0).raw() & (1 << 14);
    const bool linkError = recdTp.sample(0).raw() & (1 << 15);

    if (towerMasked) {
      eventMonitors.hcalOccTowerMasked_->Fill(ieta, iphi);
    }
    if (linkMasked) {
      eventMonitors.hcalOccLinkMasked_->Fill(ieta, iphi);
    }

    if (sentTp.SOI_compressedEt() > tpFillThreshold_) {
      eventMonitors.hcalTPRawEtSent_->Fill(sentTp.SOI_compressedEt());
      eventMonitors.hcalOccSent_->Fill(ieta, iphi);
    }

    // 6 HCAL fine grain bits from uHTR readout
    bool uHTRfg0 = sentTp.SOI_fineGrain(0);
    bool uHTRfg1 = sentTp.SOI_fineGrain(1);
    bool uHTRfg2 = sentTp.SOI_fineGrain(2);
    bool uHTRfg3 = sentTp.SOI_fineGrain(3);
    bool uHTRfg4 = sentTp.SOI_fineGrain(4);
    bool uHTRfg5 = sentTp.SOI_fineGrain(5);

    if (uHTRfg0) {
      eventMonitors.hcalOccSentFg0_->Fill(ieta, iphi);
    }
    if (uHTRfg1) {
      eventMonitors.hcalOccSentFg1_->Fill(ieta, iphi);
    }
    if (uHTRfg2) {
      eventMonitors.hcalOccSentFg2_->Fill(ieta, iphi);
    }
    if (uHTRfg3) {
      eventMonitors.hcalOccSentFg3_->Fill(ieta, iphi);
    }
    if (uHTRfg4) {
      eventMonitors.hcalOccSentFg4_->Fill(ieta, iphi);
    }
    if (uHTRfg5) {
      eventMonitors.hcalOccSentFg5_->Fill(ieta, iphi);
    }

    if (towerMasked || caloLayer1OutOfRun) {
      // Do not compare if we have a mask applied
      continue;
    }

    if (linkError) {
      eventMonitors.hcalLinkError_->Fill(ieta, iphi);
      eventMonitors.hcalLinkErrorByLumi_->Fill(event.id().luminosityBlock());
      nHcalLinkErrors++;
      // Don't compare anymore, we already know its bad
      continue;
    }

    // 6 HCAL fine grain bits from Layer1 readout
    // Fg 0 is always there in ctp7 payload, set as bit 8 after Et bits in unpacked towerDatum
    // When additionalFB flag is set:
    // Fg 1-5 are unpacked and set as bits 0-5 in another unpacked towerDatum2 and packed in HCALTP sample(1)
    // since standard sample size is 16bit and there is no room in sample(0) which contains already Et and link status
    // Otherwise:
    // Fg 1-5 are all zero
    bool layer1fg0 = recdTp.SOI_fineGrain(0);
    bool layer1fg1 = false;
    bool layer1fg2 = false;
    bool layer1fg3 = false;
    bool layer1fg4 = false;
    bool layer1fg5 = false;
    if (additionalFB && (abs(ieta) < 29)) {
      for (const auto& tp : (*hcalTPsRecd)) {
        if (not(tp.id().ieta() == ieta && tp.id().iphi() == iphi)) {
          continue;
        }
        layer1fg1 = tp.sample(1).raw() & (1 << 0);
        layer1fg2 = tp.sample(1).raw() & (1 << 1);
        layer1fg3 = tp.sample(1).raw() & (1 << 2);
        layer1fg4 = tp.sample(1).raw() & (1 << 3);
        layer1fg5 = tp.sample(1).raw() & (1 << 4);
      }
    }

    // Check mismatches only for HBHE
    const bool Hfg0Agreement = (abs(ieta) < 29) ? (layer1fg0 == uHTRfg0) : true;
    const bool Hfg1Agreement = (abs(ieta) < 29) ? (layer1fg1 == uHTRfg1) : true;
    const bool Hfg2Agreement = (abs(ieta) < 29) ? (layer1fg2 == uHTRfg2) : true;
    const bool Hfg3Agreement = (abs(ieta) < 29) ? (layer1fg3 == uHTRfg3) : true;
    const bool Hfg4Agreement = (abs(ieta) < 29) ? (layer1fg4 == uHTRfg4) : true;
    const bool Hfg5Agreement = (abs(ieta) < 29) ? (layer1fg5 == uHTRfg5) : true;
    // Mute fg4 and fg5 for now (reserved bits not used anyway)
    const bool HfgAgreement = (Hfg0Agreement && Hfg1Agreement && Hfg2Agreement && Hfg3Agreement);

    // Construct an 6-bit integer from the layer1 fine grain readout (input to 6:1 logic emulation)
    uint64_t fg_bits = 0;
    if (layer1fg0) {
      fg_bits |= 0x1;
    }
    if (layer1fg1) {
      fg_bits |= 0x1 << 1;
    }
    if (layer1fg2) {
      fg_bits |= 0x1 << 2;
    }
    if (layer1fg3) {
      fg_bits |= 0x1 << 3;
    }
    if (layer1fg4) {
      fg_bits |= 0x1 << 4;
    }
    if (layer1fg5) {
      fg_bits |= 0x1 << 5;
    }

    // Current 6:1 LUT in fw
    const uint64_t HCalFbLUT = 0xBBBABBBABBBABBBA;
    // Expected LLP bit output (mute emulation for normal events, since layer2 only reads out FAT events)
    const bool LLPfb_Expd = (FATevent == 1) ? ((HCalFbLUT >> fg_bits) & 1) : false;
    // Actual LLP bit output in layer2 data collection
    uint32_t tower_hwqual = 0;
    for (auto tower = caloTowerDataCol->begin(0); tower != caloTowerDataCol->end(0); ++tower) {
      if (not(tower->hwEta() == ieta && tower->hwPhi() == iphi)) {
        continue;
      }
      tower_hwqual = tower->hwQual();
    }
    // CaloTower hwQual is 4-bit long, LLP output bit is set at the 2nd bit (counting from 0)
    const bool LLPfb_Data = ((tower_hwqual & 0b0100) >> 2) & 1;

    const bool LLPfbAgreement = (abs(ieta) < 29) ? (LLPfb_Expd == LLPfb_Data) : true;

    // Fill feature bits Occ for HBHE only
    if (abs(ieta) < 29) {
      if (layer1fg0) {
        eventMonitors.hcalOccRecdFg0_->Fill(ieta, iphi);
      }
      if (layer1fg1) {
        eventMonitors.hcalOccRecdFg1_->Fill(ieta, iphi);
      }
      if (layer1fg2) {
        eventMonitors.hcalOccRecdFg2_->Fill(ieta, iphi);
      }
      if (layer1fg3) {
        eventMonitors.hcalOccRecdFg3_->Fill(ieta, iphi);
      }
      if (layer1fg4) {
        eventMonitors.hcalOccRecdFg4_->Fill(ieta, iphi);
      }
      if (layer1fg5) {
        eventMonitors.hcalOccRecdFg5_->Fill(ieta, iphi);
      }
      // fg4-5 are reserved bits and not used
      // so compare here and not stream to mismatch list for now
      if (not Hfg4Agreement) {
        eventMonitors.hcalOccFg4Discrepancy_->Fill(ieta, iphi);
      }
      if (not Hfg5Agreement) {
        eventMonitors.hcalOccFg5Discrepancy_->Fill(ieta, iphi);
      }
      // Fill Fb Occ and compare between layer1 emulated and layer2 data readout
      // FAT events only!!
      if (LLPfb_Expd) {
        eventMonitors.hcalOccLLPFbExpd_->Fill(ieta, iphi);
      }
      if (LLPfb_Data) {
        eventMonitors.hcalOccLLPFbData_->Fill(ieta, iphi);
      }
    }

    if (recdTp.SOI_compressedEt() > tpFillThreshold_) {
      eventMonitors.hcalTPRawEtRecd_->Fill(recdTp.SOI_compressedEt());
      eventMonitors.hcalOccupancy_->Fill(ieta, iphi);
      eventMonitors.hcalOccRecdEtWgt_->Fill(ieta, iphi, recdTp.SOI_compressedEt());
    }

    if (abs(ieta) > 29) {
      eventMonitors.hcalTPRawEtCorrelationHF_->Fill(sentTp.SOI_compressedEt(), recdTp.SOI_compressedEt());
    } else {
      eventMonitors.hcalTPRawEtCorrelationHBHE_->Fill(sentTp.SOI_compressedEt(), recdTp.SOI_compressedEt());
    }

    const bool HetAgreement = sentTp.SOI_compressedEt() == recdTp.SOI_compressedEt();
    if (HetAgreement && HfgAgreement && LLPfbAgreement) {
      // Full match
      if (sentTp.SOI_compressedEt() > tpFillThreshold_) {
        eventMonitors.hcalOccSentAndRecd_->Fill(ieta, iphi);
        eventMonitors.hcalTPRawEtSentAndRecd_->Fill(sentTp.SOI_compressedEt());
      }
    } else {
      // There is some issue
      eventMonitors.hcalDiscrepancy_->Fill(ieta, iphi);
      eventMonitors.hcalMismatchByLumi_->Fill(event.id().luminosityBlock());
      nHcalMismatch++;

      if (not HetAgreement) {
        if (abs(ieta) > 29) {
          eventMonitors.HFmismatchesPerBx_->Fill(event.bunchCrossing());
        } else {
          eventMonitors.HBHEmismatchesPerBx_->Fill(event.bunchCrossing());
        }
        eventMonitors.hcalOccEtDiscrepancy_->Fill(ieta, iphi);
        eventMonitors.hcalTPRawEtDiffNoMatch_->Fill(recdTp.SOI_compressedEt() - sentTp.SOI_compressedEt());
        updateMismatch(event, 2, streamCache(event.streamID())->streamMismatchList);

        // Handle HCal discrepancy debug
        if (sentTp.SOI_compressedEt() == 0)
          eventMonitors.hcalOccRecdNotSent_->Fill(ieta, iphi);
        else if (recdTp.SOI_compressedEt() == 0)
          eventMonitors.hcalOccSentNotRecd_->Fill(ieta, iphi);
        else
          eventMonitors.hcalOccNoMatch_->Fill(ieta, iphi);
      }
      if (not(HfgAgreement && LLPfbAgreement)) {
        if (not Hfg0Agreement) {
          eventMonitors.hcalOccFg0Discrepancy_->Fill(ieta, iphi);
        }
        if (not Hfg1Agreement) {
          eventMonitors.hcalOccFg1Discrepancy_->Fill(ieta, iphi);
        }
        if (not Hfg2Agreement) {
          eventMonitors.hcalOccFg2Discrepancy_->Fill(ieta, iphi);
        }
        if (not Hfg3Agreement) {
          eventMonitors.hcalOccFg3Discrepancy_->Fill(ieta, iphi);
        }
        if (not LLPfbAgreement) {
          eventMonitors.hcalOccLLPFbDiscrepancy_->Fill(ieta, iphi);
        }
        updateMismatch(event, 3, streamCache(event.streamID())->streamMismatchList);
      }
    }
  }

  if (nHcalLinkErrors > streamCache(event.streamID())->streamNumMaxEvtLinkErrorsHCAL)
    streamCache(event.streamID())->streamNumMaxEvtLinkErrorsHCAL = nHcalLinkErrors;
  if (nHcalMismatch > streamCache(event.streamID())->streamNumMaxEvtMismatchHCAL)
    streamCache(event.streamID())->streamNumMaxEvtMismatchHCAL = nHcalMismatch;

  //fill inclusive link error and mismatch cache values based on whether HCAL or ECAL had more this event
  if (nEcalLinkErrors >= nHcalLinkErrors && nEcalLinkErrors > streamCache(event.streamID())->streamNumMaxEvtLinkErrors)
    streamCache(event.streamID())->streamNumMaxEvtLinkErrors = nEcalLinkErrors;
  else if (nEcalLinkErrors < nHcalLinkErrors &&
           nHcalLinkErrors > streamCache(event.streamID())->streamNumMaxEvtLinkErrors)
    streamCache(event.streamID())->streamNumMaxEvtLinkErrors = nHcalLinkErrors;

  if (nEcalMismatch >= nHcalMismatch && nEcalMismatch > streamCache(event.streamID())->streamNumMaxEvtMismatch)
    streamCache(event.streamID())->streamNumMaxEvtMismatch = nEcalMismatch;
  else if (nEcalMismatch < nHcalMismatch && nHcalMismatch > streamCache(event.streamID())->streamNumMaxEvtMismatch)
    streamCache(event.streamID())->streamNumMaxEvtMismatch = nHcalMismatch;
}

//push the mismatch type and identifying information onto the back of the stream-based vector
//will maintain the vector's size at 20
void L1TStage2CaloLayer1::updateMismatch(
    const edm::Event& e,
    int mismatchType,
    std::vector<std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>>>& streamMismatches)
    const {
  //check if this combination of Run/Lumi/Event already exists in the stream mismatch list
  //if it does, update that entry, otherwise insert a new one with this particular combination.
  for (auto mismatchIterator = streamMismatches.begin(); mismatchIterator != streamMismatches.end();
       ++mismatchIterator) {
    if (e.getRun().id() == std::get<0>(*mismatchIterator) &&
        e.getLuminosityBlock().id() == std::get<1>(*mismatchIterator) && e.id() == std::get<2>(*mismatchIterator)) {
      //the run, lumi and event exist. Check if this kind of mismatch has been reported before
      std::vector<int>& mismatchTypeVector = std::get<3>(*mismatchIterator);
      for (auto mismatchTypeIterator = mismatchTypeVector.begin(); mismatchTypeIterator != mismatchTypeVector.end();
           ++mismatchTypeIterator) {
        if (mismatchType == *mismatchTypeIterator) {
          //this has already been reported
          return;
        }
      }
      //A mismatch exists, but it is not a type that has been previously reported.
      //Insert it into the vector of types of mismatches reported
      mismatchTypeVector.push_back(mismatchType);
      return;
    }
  }

  //The run/lumi/event does not exist in the list, construct an entry
  std::vector<int> newMismatchTypeVector;
  newMismatchTypeVector.push_back(mismatchType);
  std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>> mismatchToInsert = {
      e.getRun().id(), e.getLuminosityBlock().id(), e.id(), newMismatchTypeVector};
  streamMismatches.push_back(mismatchToInsert);

  //maintain the mismatch vector size at 20.
  if (streamMismatches.size() > 20)
    streamMismatches.erase(streamMismatches.begin());
}

void L1TStage2CaloLayer1::dqmBeginRun(const edm::Run&,
                                      const edm::EventSetup&,
                                      CaloL1Information::monitoringDataHolder& eventMonitors) const {}

void L1TStage2CaloLayer1::dqmEndRun(const edm::Run& run,
                                    const edm::EventSetup& es,
                                    const CaloL1Information::monitoringDataHolder& runMonitors,
                                    const CaloL1Information::perRunSummaryMonitoringInformation&) const {}

void L1TStage2CaloLayer1::bookHistograms(DQMStore::IBooker& ibooker,
                                         const edm::Run& run,
                                         const edm::EventSetup& es,
                                         CaloL1Information::monitoringDataHolder& eventMonitors) const {
  auto bookEt = [&ibooker](std::string name, std::string title) {
    return ibooker.book1D(name, title + ";Raw ET;Counts", 256, -0.5, 255.5);
  };
  auto bookEtCorrelation = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title, 256, -0.5, 255.5, 256, -0.5, 255.5);
  };
  auto bookEtDiff = [&ibooker](std::string name, std::string title) {
    return ibooker.book1D(name, title + ";#Delta Raw ET;Counts", 511, -255.5, 255.5);
  };
  auto bookEcalOccupancy = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title + ";iEta;iPhi", 57, -28.5, 28.5, 72, 0.5, 72.5);
  };
  auto bookHcalOccupancy = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title + ";iEta;iPhi", 83, -41.5, 41.5, 72, 0.5, 72.5);
  };

  ibooker.setCurrentFolder(histFolder_);

  eventMonitors.ecalDiscrepancy_ =
      bookEcalOccupancy("ecalDiscrepancy", "ECAL Discrepancies between TCC and Layer1 Readout");
  eventMonitors.ecalLinkError_ = bookEcalOccupancy("ecalLinkError", "ECAL Link Errors");
  eventMonitors.ecalOccupancy_ = bookEcalOccupancy("ecalOccupancy", "ECAL TP Occupancy at Layer1");
  eventMonitors.ecalOccRecdEtWgt_ = bookEcalOccupancy("ecalOccRecdEtWgt", "ECal TP ET-weighted Occupancy at Layer1");
  eventMonitors.hcalDiscrepancy_ =
      bookHcalOccupancy("hcalDiscrepancy", "HCAL Discrepancies between uHTR and Layer1 Readout");
  eventMonitors.hcalLinkError_ = bookHcalOccupancy("hcalLinkError", "HCAL Link Errors");
  eventMonitors.hcalOccupancy_ = bookHcalOccupancy("hcalOccupancy", "HCAL TP Occupancy at Layer1");
  eventMonitors.hcalOccRecdEtWgt_ = bookHcalOccupancy("hcalOccRecdEtWgt", "HCal TP ET-weighted Occupancy at Layer1");

  ibooker.setCurrentFolder(histFolder_ + "/ECalDetail");

  eventMonitors.ecalOccEtDiscrepancy_ = bookEcalOccupancy("ecalOccEtDiscrepancy", "ECal Et Discrepancy Occupancy");
  eventMonitors.ecalOccFgDiscrepancy_ =
      bookEcalOccupancy("ecalOccFgDiscrepancy", "ECal FG Veto Bit Discrepancy Occupancy");
  eventMonitors.ecalOccLinkMasked_ = bookEcalOccupancy("ecalOccLinkMasked", "ECal Masked Links");
  eventMonitors.ecalOccRecdFgVB_ = bookEcalOccupancy("ecalOccRecdFgVB", "ECal FineGrain Veto Bit Occupancy at Layer1");
  eventMonitors.ecalOccSentAndRecd_ = bookEcalOccupancy("ecalOccSentAndRecd", "ECal TP Occupancy FULL MATCH");
  eventMonitors.ecalOccSentFgVB_ = bookEcalOccupancy("ecalOccSentFgVB", "ECal FineGrain Veto Bit Occupancy at TCC");
  eventMonitors.ecalOccSent_ = bookEcalOccupancy("ecalOccSent", "ECal TP Occupancy at TCC");
  eventMonitors.ecalOccTowerMasked_ = bookEcalOccupancy("ecalOccTowerMasked", "ECal Masked towers");
  eventMonitors.ecalTPRawEtCorrelation_ =
      bookEtCorrelation("ecalTPRawEtCorrelation", "Raw Et correlation TCC and Layer1;TCC Et;Layer1 Et");
  eventMonitors.ecalTPRawEtDiffNoMatch_ = bookEtDiff("ecalTPRawEtDiffNoMatch", "ECal Raw Et Difference Layer1 - TCC");
  eventMonitors.ecalTPRawEtRecd_ = bookEt("ecalTPRawEtRecd", "ECal Raw Et Layer1 Readout");
  eventMonitors.ecalTPRawEtSentAndRecd_ = bookEt("ecalTPRawEtMatch", "ECal Raw Et FULL MATCH");
  eventMonitors.ecalTPRawEtSent_ = bookEt("ecalTPRawEtSent", "ECal Raw Et TCC Readout");
  eventMonitors.ecalOccRecd5Bx_ = ibooker.book1D("ecalOccRecd5Bx", "Number of TPs vs BX", 5, 1, 6);
  eventMonitors.ecalOccRecd5BxEtWgt_ =
      ibooker.book1D("ecalOccRecd5BxEtWgt", "Et-weighted number of TPs vs BX", 5, 1, 6);
  eventMonitors.ecalOccRecdBx1_ = bookEcalOccupancy("ecalOccRecdBx1", "ECal TP Occupancy for BX1");
  eventMonitors.ecalOccRecdBx2_ = bookEcalOccupancy("ecalOccRecdBx2", "ECal TP Occupancy for BX2");
  eventMonitors.ecalOccRecdBx3_ = bookEcalOccupancy("ecalOccRecdBx3", "ECal TP Occupancy for BX3");
  eventMonitors.ecalOccRecdBx4_ = bookEcalOccupancy("ecalOccRecdBx4", "ECal TP Occupancy for BX4");
  eventMonitors.ecalOccRecdBx5_ = bookEcalOccupancy("ecalOccRecdBx5", "ECal TP Occupancy for BX5");

  ibooker.setCurrentFolder(histFolder_ + "/ECalDetail/TCCDebug");
  eventMonitors.ecalOccSentNotRecd_ =
      bookHcalOccupancy("ecalOccSentNotRecd", "ECal TP Occupancy sent by TCC, zero at Layer1");
  eventMonitors.ecalOccRecdNotSent_ =
      bookHcalOccupancy("ecalOccRecdNotSent", "ECal TP Occupancy received by Layer1, zero at TCC");
  eventMonitors.ecalOccNoMatch_ =
      bookHcalOccupancy("ecalOccNoMatch", "ECal TP Occupancy for TCC and Layer1 nonzero, not matching");

  ibooker.setCurrentFolder(histFolder_ + "/HCalDetail");

  eventMonitors.hcalOccEtDiscrepancy_ = bookHcalOccupancy("hcalOccEtDiscrepancy", "HCal Et Discrepancy Occupancy");
  eventMonitors.hcalOccLLPFbDiscrepancy_ =
      bookHcalOccupancy("hcalOccLLPFbDiscrepancy", "HCal LLP Feature Bit Discrepancy between Expected and Data");
  eventMonitors.hcalOccLLPFbExpd_ = bookHcalOccupancy("hcalOccLLPFbExpd", "HCal LLP Feature Bit Occupancy Expected");
  eventMonitors.hcalOccLLPFbData_ = bookHcalOccupancy("hcalOccLLPFbData", "HCal LLP Feature Bit Occupancy in Data");
  eventMonitors.hcalOccFg0Discrepancy_ =
      bookHcalOccupancy("hcalOccFg0Discrepancy", "HCal Fine Grain 0 Discrepancy between uHTR and Layer1");
  eventMonitors.hcalOccFg1Discrepancy_ =
      bookHcalOccupancy("hcalOccFg1Discrepancy", "HCal Fine Grain 1 Discrepancy between uHTR and Layer1");
  eventMonitors.hcalOccFg2Discrepancy_ =
      bookHcalOccupancy("hcalOccFg2Discrepancy", "HCal Fine Grain 2 Discrepancy between uHTR and Layer1");
  eventMonitors.hcalOccFg3Discrepancy_ =
      bookHcalOccupancy("hcalOccFg3Discrepancy", "HCal Fine Grain 3 Discrepancy between uHTR and Layer1");
  eventMonitors.hcalOccFg4Discrepancy_ =
      bookHcalOccupancy("hcalOccFg4Discrepancy", "HCal Fine Grain 4 Discrepancy between uHTR and Layer1");
  eventMonitors.hcalOccFg5Discrepancy_ =
      bookHcalOccupancy("hcalOccFg5Discrepancy", "HCal Fine Grain 5 Discrepancy between uHTR and Layer1");
  eventMonitors.hcalOccSentFg0_ = bookHcalOccupancy("hcalOccSentFg0", "HCal Fine Grain 0 Occupancy at uHTR");
  eventMonitors.hcalOccSentFg1_ = bookHcalOccupancy("hcalOccSentFg1", "HCal Fine Grain 1 Occupancy at uHTR");
  eventMonitors.hcalOccSentFg2_ = bookHcalOccupancy("hcalOccSentFg2", "HCal Fine Grain 2 Occupancy at uHTR");
  eventMonitors.hcalOccSentFg3_ = bookHcalOccupancy("hcalOccSentFg3", "HCal Fine Grain 3 Occupancy at uHTR");
  eventMonitors.hcalOccSentFg4_ = bookHcalOccupancy("hcalOccSentFg4", "HCal Fine Grain 4 Occupancy at uHTR");
  eventMonitors.hcalOccSentFg5_ = bookHcalOccupancy("hcalOccSentFg5", "HCal Fine Grain 5 Occupancy at uHTR");
  eventMonitors.hcalOccRecdFg0_ = bookHcalOccupancy("hcalOccRecdFg0", "HCal Fine Grain 0 Occupancy at Layer1");
  eventMonitors.hcalOccRecdFg1_ = bookHcalOccupancy("hcalOccRecdFg1", "HCal Fine Grain 1 Occupancy at Layer1");
  eventMonitors.hcalOccRecdFg2_ = bookHcalOccupancy("hcalOccRecdFg2", "HCal Fine Grain 2 Occupancy at Layer1");
  eventMonitors.hcalOccRecdFg3_ = bookHcalOccupancy("hcalOccRecdFg3", "HCal Fine Grain 3 Occupancy at Layer1");
  eventMonitors.hcalOccRecdFg4_ = bookHcalOccupancy("hcalOccRecdFg4", "HCal Fine Grain 4 Occupancy at Layer1");
  eventMonitors.hcalOccRecdFg5_ = bookHcalOccupancy("hcalOccRecdFg5", "HCal Fine Grain 5 Occupancy at Layer1");
  eventMonitors.hcalOccLinkMasked_ = bookHcalOccupancy("hcalOccLinkMasked", "HCal Masked Links");
  eventMonitors.hcalOccSentAndRecd_ = bookHcalOccupancy("hcalOccSentAndRecd", "HCal TP Occupancy FULL MATCH");
  eventMonitors.hcalOccSent_ = bookHcalOccupancy("hcalOccSent", "HCal TP Occupancy at uHTR");
  eventMonitors.hcalOccTowerMasked_ = bookHcalOccupancy("hcalOccTowerMasked", "HCal Masked towers");
  eventMonitors.hcalTPRawEtCorrelationHBHE_ =
      bookEtCorrelation("hcalTPRawEtCorrelationHBHE", "HBHE Raw Et correlation uHTR and Layer1;uHTR Et;Layer1 Et");
  eventMonitors.hcalTPRawEtCorrelationHF_ =
      bookEtCorrelation("hcalTPRawEtCorrelationHF", "HF Raw Et correlation uHTR and Layer1;uHTR Et;Layer1 Et");
  eventMonitors.hcalTPRawEtDiffNoMatch_ = bookEtDiff("hcalTPRawEtDiffNoMatch", "HCal Raw Et Difference Layer1 - uHTR");
  eventMonitors.hcalTPRawEtRecd_ = bookEt("hcalTPRawEtRecd", "HCal Raw Et Layer1 Readout");
  eventMonitors.hcalTPRawEtSentAndRecd_ = bookEt("hcalTPRawEtMatch", "HCal Raw Et FULL MATCH");
  eventMonitors.hcalTPRawEtSent_ = bookEt("hcalTPRawEtSent", "HCal Raw Et uHTR Readout");

  ibooker.setCurrentFolder(histFolder_ + "/HCalDetail/uHTRDebug");
  eventMonitors.hcalOccSentNotRecd_ =
      bookHcalOccupancy("hcalOccSentNotRecd", "HCal TP Occupancy sent by uHTR, zero at Layer1");
  eventMonitors.hcalOccRecdNotSent_ =
      bookHcalOccupancy("hcalOccRecdNotSent", "HCal TP Occupancy received by Layer1, zero at uHTR");
  eventMonitors.hcalOccNoMatch_ =
      bookHcalOccupancy("hcalOccNoMatch", "HCal TP Occupancy for uHTR and Layer1 nonzero, not matching");

  ibooker.setCurrentFolder(histFolder_ + "/MismatchDetail");

  const int nMismatchTypes = 4;
  eventMonitors.last20Mismatches_ = ibooker.book2D("last20Mismatches",
                                                   "Log of last 20 mismatches (use json tool to copy/paste)",
                                                   nMismatchTypes,
                                                   0,
                                                   nMismatchTypes,
                                                   20,
                                                   0,
                                                   20);
  eventMonitors.last20Mismatches_->setBinLabel(1, "Ecal TP Et Mismatch");
  eventMonitors.last20Mismatches_->setBinLabel(2, "Ecal TP Fine Grain Bit Mismatch");
  eventMonitors.last20Mismatches_->setBinLabel(3, "Hcal TP Et Mismatch");
  eventMonitors.last20Mismatches_->setBinLabel(4, "Hcal TP Feature Bit Mismatch");

  const int nLumis = 2000;
  eventMonitors.ecalLinkErrorByLumi_ = ibooker.book1D(
      "ecalLinkErrorByLumi", "Link error counts per lumi section for ECAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  eventMonitors.ecalMismatchByLumi_ = ibooker.book1D(
      "ecalMismatchByLumi", "Mismatch counts per lumi section for ECAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  eventMonitors.hcalLinkErrorByLumi_ = ibooker.book1D(
      "hcalLinkErrorByLumi", "Link error counts per lumi section for HCAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  eventMonitors.hcalMismatchByLumi_ = ibooker.book1D(
      "hcalMismatchByLumi", "Mismatch counts per lumi section for HCAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);

  eventMonitors.ECALmismatchesPerBx_ =
      ibooker.book1D("ECALmismatchesPerBx", "Mismatch counts per bunch crossing for ECAL", 3564, -.5, 3563.5);
  eventMonitors.HBHEmismatchesPerBx_ =
      ibooker.book1D("HBHEmismatchesPerBx", "Mismatch counts per bunch crossing for HBHE", 3564, -.5, 3563.5);
  eventMonitors.HFmismatchesPerBx_ =
      ibooker.book1D("HFmismatchesPerBx", "Mismatch counts per bunch crossing for HF", 3564, -.5, 3563.5);

  eventMonitors.maxEvtLinkErrorsByLumiECAL_ =
      ibooker.book1D("maxEvtLinkErrorsByLumiECAL",
                     "Max number of single-event ECAL link errors per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);
  eventMonitors.maxEvtLinkErrorsByLumiHCAL_ =
      ibooker.book1D("maxEvtLinkErrorsByLumiHCAL",
                     "Max number of single-event HCAL link errors per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);

  eventMonitors.maxEvtMismatchByLumiECAL_ =
      ibooker.book1D("maxEvtMismatchByLumiECAL",
                     "Max number of single-event ECAL discrepancies per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);
  eventMonitors.maxEvtMismatchByLumiHCAL_ =
      ibooker.book1D("maxEvtMismatchByLumiHCAL",
                     "Max number of single-event HCAL discrepancies per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);

  ibooker.setCurrentFolder(histFolder_);
  eventMonitors.maxEvtLinkErrorsByLumi_ =
      ibooker.book1D("maxEvtLinkErrorsByLumi",
                     "Max number of single-event link errors per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);
  eventMonitors.maxEvtMismatchByLumi_ =
      ibooker.book1D("maxEvtMismatchByLumi",
                     "Max number of single-event discrepancies per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);

  ibooker.setCurrentFolder(histFolder_ + "/AMC13ErrorCounters");
  eventMonitors.bxidErrors_ =
      ibooker.book1D("bxidErrors", "bxid mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
  eventMonitors.l1idErrors_ =
      ibooker.book1D("l1idErrors", "l1id mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
  eventMonitors.orbitErrors_ =
      ibooker.book1D("orbitErrors", "orbit mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
}

void L1TStage2CaloLayer1::streamEndLuminosityBlockSummary(
    edm::StreamID theStreamID,
    edm::LuminosityBlock const& theLumiBlock,
    edm::EventSetup const& theEventSetup,
    CaloL1Information::perLumiBlockMonitoringInformation* lumiMonitoringInformation) const {
  auto theStreamCache = streamCache(theStreamID);

  lumiMonitoringInformation->lumiNumMaxEvtLinkErrorsECAL =
      std::max(lumiMonitoringInformation->lumiNumMaxEvtLinkErrorsECAL, theStreamCache->streamNumMaxEvtLinkErrorsECAL);
  lumiMonitoringInformation->lumiNumMaxEvtLinkErrorsHCAL =
      std::max(lumiMonitoringInformation->lumiNumMaxEvtLinkErrorsHCAL, theStreamCache->streamNumMaxEvtLinkErrorsHCAL);
  lumiMonitoringInformation->lumiNumMaxEvtLinkErrors =
      std::max(lumiMonitoringInformation->lumiNumMaxEvtLinkErrors, theStreamCache->streamNumMaxEvtLinkErrors);

  lumiMonitoringInformation->lumiNumMaxEvtMismatchECAL =
      std::max(lumiMonitoringInformation->lumiNumMaxEvtMismatchECAL, theStreamCache->streamNumMaxEvtMismatchECAL);
  lumiMonitoringInformation->lumiNumMaxEvtMismatchHCAL =
      std::max(lumiMonitoringInformation->lumiNumMaxEvtMismatchHCAL, theStreamCache->streamNumMaxEvtMismatchHCAL);
  lumiMonitoringInformation->lumiNumMaxEvtMismatch =
      std::max(lumiMonitoringInformation->lumiNumMaxEvtMismatch, theStreamCache->streamNumMaxEvtMismatch);

  //reset the stream cache here, since we are done with the luminosity block in this stream
  //We don't want the stream to be comparing to previous values in the next lumi-block
  theStreamCache->streamNumMaxEvtLinkErrorsECAL = 0;
  theStreamCache->streamNumMaxEvtLinkErrorsHCAL = 0;
  theStreamCache->streamNumMaxEvtLinkErrors = 0;

  theStreamCache->streamNumMaxEvtMismatchECAL = 0;
  theStreamCache->streamNumMaxEvtMismatchHCAL = 0;
  theStreamCache->streamNumMaxEvtMismatch = 0;
}

void L1TStage2CaloLayer1::globalEndLuminosityBlockSummary(
    edm::LuminosityBlock const& theLumiBlock,
    edm::EventSetup const& theEventSetup,
    CaloL1Information::perLumiBlockMonitoringInformation* lumiMonitoringInformation) const {
  auto theRunCache = runCache(theLumiBlock.getRun().index());
  //read the cache out to the relevant Luminosity histogram bin
  theRunCache->maxEvtLinkErrorsByLumiECAL_->Fill(theLumiBlock.luminosityBlock(),
                                                 lumiMonitoringInformation->lumiNumMaxEvtLinkErrorsECAL);
  theRunCache->maxEvtLinkErrorsByLumiHCAL_->Fill(theLumiBlock.luminosityBlock(),
                                                 lumiMonitoringInformation->lumiNumMaxEvtLinkErrorsHCAL);
  theRunCache->maxEvtLinkErrorsByLumi_->Fill(theLumiBlock.luminosityBlock(),
                                             lumiMonitoringInformation->lumiNumMaxEvtLinkErrors);

  theRunCache->maxEvtMismatchByLumiECAL_->Fill(theLumiBlock.luminosityBlock(),
                                               lumiMonitoringInformation->lumiNumMaxEvtMismatchECAL);
  theRunCache->maxEvtMismatchByLumiHCAL_->Fill(theLumiBlock.luminosityBlock(),
                                               lumiMonitoringInformation->lumiNumMaxEvtMismatchHCAL);
  theRunCache->maxEvtMismatchByLumi_->Fill(theLumiBlock.luminosityBlock(),
                                           lumiMonitoringInformation->lumiNumMaxEvtMismatch);

  // Simple way to embed current lumi to auto-scale axis limits in render plugin
  theRunCache->ecalLinkErrorByLumi_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->ecalMismatchByLumi_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->hcalLinkErrorByLumi_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->hcalMismatchByLumi_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->maxEvtLinkErrorsByLumiECAL_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->maxEvtLinkErrorsByLumiHCAL_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->maxEvtLinkErrorsByLumi_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->maxEvtMismatchByLumiECAL_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->maxEvtMismatchByLumiHCAL_->setBinContent(0, theLumiBlock.luminosityBlock());
  theRunCache->maxEvtMismatchByLumi_->setBinContent(0, theLumiBlock.luminosityBlock());
}

//returns true if the new candidate mismatch is from a later mismatch than the comparison mismatch
// based on Run, Lumisection, and event number
//false otherwise
bool L1TStage2CaloLayer1::isLaterMismatch(
    std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>>& candidateMismatch,
    std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>>& comparisonMismatch) const {
  //check the run. If the run ID of the candidate mismatch is less than the run ID of the comparison mismatch, it is earlier,
  if (std::get<0>(candidateMismatch) < std::get<0>(comparisonMismatch))
    return false;
  //if it is greater, then it is a later mismatch
  else if (std::get<0>(candidateMismatch) > std::get<0>(comparisonMismatch))
    return true;
  //if it is even, then we need to repeat this comparison on the luminosity block
  else {
    if (std::get<1>(candidateMismatch) < std::get<1>(comparisonMismatch))
      return false;  //if the lumi block is less, it is an earlier mismatch
    else if (std::get<1>(candidateMismatch) > std::get<1>(comparisonMismatch))
      return true;  // if the lumi block is greater, than it is a later mismatch
    // if these are equivalent, then we repeat the comparison on the event
    else {
      if (std::get<2>(candidateMismatch) < std::get<2>(comparisonMismatch))
        return false;
      else
        return true;  //in the case of even events here, we consider the event later.
    }
  }
  //default return, should never be called.
  return false;
}

//binary search like algorithm for trying to find the appropriate place to put our new mismatch
//will find an interger we add to the iterator to get the proper location to find the insertion location
//will return -1 if the mismatch should not be inserted into the list
int L1TStage2CaloLayer1::findIndex(
    std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>> candidateMismatch,
    std::vector<std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>>> comparisonList,
    int lowerIndexToSearch,
    int upperIndexToSearch) const {
  //Start by getting the spot in the the vector to start searching
  int searchLocation = ((upperIndexToSearch + lowerIndexToSearch) / 2);
  auto searchIterator = comparisonList.begin() + searchLocation;
  //Multiple possible cases:
  //case one. Greater than the search element
  if (this->isLaterMismatch(candidateMismatch, *searchIterator)) {
    //subcase one, there exists a mismatch to its right,
    if (searchIterator + 1 != comparisonList.end()) {
      //subsubcase one, the candidate is earlier than the one to the right, insert this element between these two
      if (not this->isLaterMismatch(candidateMismatch, *(searchIterator + 1))) {
        return searchLocation + 1;  //vector insert inserts *before* the position, so we return +1
      }
      //subsubcase two, the candidate is later than it, in this case we refine the search area.
      else {
        return this->findIndex(candidateMismatch, comparisonList, searchLocation + 1, upperIndexToSearch);
      }
    }
    //subcase two, there exists no mismatch to it's right (end of the vector), in which case this is the latest mismatch
    else {
      return searchLocation +
             1;  //vector insert inserts *before* the position, so we return +1 (should be synonymous with .end())
    }
  }
  //case two. we are earlier than the current mismatch
  else {
    //subcase one, these exists a mismatch to its left,
    if (searchIterator != comparisonList.begin()) {
      //subsubcase one, the candidate mismatch is earlier than the one to the left. in this case we refine the search area
      if (not this->isLaterMismatch(candidateMismatch, *(searchIterator - 1))) {
        return this->findIndex(candidateMismatch, comparisonList, lowerIndexToSearch, searchLocation - 1);
      }
      //subsubcase two the candidate is later than than the one to it's left, in which case, we insert the element between these two.
      else {
        return searchLocation;  //vector insert inserts *before* the position, so we return without addition here.
      }
    }
    //subcase two, there exists no mismatch to its left (beginning of vector), in which case this is the is earlier than all mismatches on the list.
    else {
      //subsubcase one. It is possible we still insert this if the capacity of the vector is below 20.
      //this should probably be rewritten to have the capacity reserved before-hand so that 20 is not potentially a hard coded magic number
      //defined by what we know to be true about a non-obvious data type elsewhere.
      if (comparisonList.size() < 20) {
        return searchLocation;
      }
      //subsubcase two. The size is 20 or above, and we are earlier than all of them. This should not be inserted into the list.
      else {
        return -1;
      }
    }
  }
  //default return. Should never be called
  return -1;
}

//will shuffle the candidate mismatch list into the comparison mismatch list.
void L1TStage2CaloLayer1::mergeMismatchVectors(
    std::vector<std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>>>& candidateMismatchList,
    std::vector<std::tuple<edm::RunID, edm::LuminosityBlockID, edm::EventID, std::vector<int>>>& comparisonMismatchList)
    const {
  //okay now we loop over our candidate mismatches
  for (auto candidateIterator = candidateMismatchList.begin(); candidateIterator != candidateMismatchList.end();
       ++candidateIterator) {
    int insertionIndex = this->findIndex(*candidateIterator, comparisonMismatchList, 0, comparisonMismatchList.size());
    if (insertionIndex < 0) {
      continue;  //if we didn't find anywhere to put this mismatch, we move on
    }
    auto insertionIterator = comparisonMismatchList.begin() + insertionIndex;
    comparisonMismatchList.insert(insertionIterator, *candidateIterator);
    //now if we have more than 20 mismatches in the list, we erase the earliest one (the beginning)
    //this should probably be rewritten to have the capacity reserved before-hand so that 20 is not potentially a hard coded magic number
    //defined by what we know to be true about a non-obvious data type elsewhere.
    if (comparisonMismatchList.size() > 20) {
      comparisonMismatchList.erase(comparisonMismatchList.begin());
    }
  }
}

void L1TStage2CaloLayer1::streamEndRunSummary(
    edm::StreamID theStreamID,
    edm::Run const& theRun,
    edm::EventSetup const& theEventSetup,
    CaloL1Information::perRunSummaryMonitoringInformation* theRunSummaryMonitoringInformation) const {
  auto theStreamCache = streamCache(theStreamID);

  if (theRunSummaryMonitoringInformation->runMismatchList.empty())
    theRunSummaryMonitoringInformation->runMismatchList = theStreamCache->streamMismatchList;
  else
    this->mergeMismatchVectors(theStreamCache->streamMismatchList, theRunSummaryMonitoringInformation->runMismatchList);

  //clear the stream cache so that the next run does not try to compare to the current one.
  theStreamCache->streamMismatchList.clear();
}

void L1TStage2CaloLayer1::globalEndRunSummary(
    edm::Run const& theRun,
    edm::EventSetup const& theEventSetup,
    CaloL1Information::perRunSummaryMonitoringInformation* theRunSummaryInformation) const {
  //The mismatch vector should be properly ordered and sized by other functions, so all we need to do is read it into a monitoring element.
  auto theRunCache = runCache(theRun.index());
  //loop over the accepted mismatch list in the run, and based on the run/lumi/event create the bin label, the mismatch type will define which bin gets content.
  int ibin = 1;
  for (auto mismatchIterator = theRunSummaryInformation->runMismatchList.begin();
       mismatchIterator != theRunSummaryInformation->runMismatchList.end();
       ++mismatchIterator) {
    std::string binLabel = std::to_string(std::get<0>(*mismatchIterator).run()) + ":" +
                           std::to_string(std::get<1>(*mismatchIterator).luminosityBlock()) + ":" +
                           std::to_string(std::get<2>(*mismatchIterator).event());
    theRunCache->last20Mismatches_->setBinLabel(ibin, binLabel, 2);
    //Get the vector of mismatches for this particular event and iterate through it.
    //Set the bin content to 1 for each type of mismatch seen
    std::vector<int> mismatchTypeVector = std::get<3>(*mismatchIterator);
    for (auto mismatchTypeIterator = mismatchTypeVector.begin(); mismatchTypeIterator != mismatchTypeVector.end();
         ++mismatchTypeIterator) {
      theRunCache->last20Mismatches_->setBinContent(*mismatchTypeIterator + 1, ibin, 1);
    }
    ++ibin;
  }
  //remove the remaining empty string labels to prevent overlap
  for (int emptyBinIndex = ibin; emptyBinIndex <= 20; ++emptyBinIndex) {
    theRunCache->last20Mismatches_->setBinLabel(emptyBinIndex, std::to_string(emptyBinIndex), 2);
  }
}
